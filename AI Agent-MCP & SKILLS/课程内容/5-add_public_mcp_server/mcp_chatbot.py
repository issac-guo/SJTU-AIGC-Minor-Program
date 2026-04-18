from dotenv import load_dotenv
import openai
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from typing import List, Dict
from contextlib import AsyncExitStack
import asyncio
import json
import os

load_dotenv()

class MCP_ChatBot:

    def __init__(self):
        # Initialize session and client objects
        self.sessions: List[ClientSession] = []
        self.exit_stack = AsyncExitStack()
        self.client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        self.available_tools: List[dict] = []
        self.tool_to_session: Dict[str, ClientSession] = {}

    async def process_query(self, query):
        messages = [{'role':'user', 'content':query}]
        response = self.client.chat.completions.create(
            model='qwen-plus',
            max_tokens=2024,
            tools=self.available_tools,
            messages=messages
        )
        
        process_query = True
        while process_query:
            # 获取助手的回复
            message = response.choices[0].message
            
            # 检查是否有普通文本内容
            if message.content:
                print(message.content)
                process_query = False
                
            # 检查是否有工具调用
            elif message.tool_calls:
                # 添加助手消息到历史
                messages.append({
                    "role": "assistant", 
                    "content": None,
                    "tool_calls": message.tool_calls
                })
                
                # 处理每个工具调用
                for tool_call in message.tool_calls:
                    tool_id = tool_call.id
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    print(f"Calling tool {tool_name} with args {tool_args}")
                    
                    # 执行工具调用
                    session = self.tool_to_session[tool_name]
                    
                    # 参数预处理：确保trainFilterFlags参数符合格式要求
                    if tool_name == "get-tickets" and "trainFilterFlags" in tool_args:
                        train_filter = tool_args["trainFilterFlags"]
                        # 确保trainFilterFlags是有效的字符串格式
                        if not isinstance(train_filter, str):
                            # 如果不是字符串，尝试转换为合适的格式
                            tool_args["trainFilterFlags"] = str(train_filter)
                        # 清理可能导致正则验证失败的特殊字符
                        tool_args["trainFilterFlags"] = tool_args["trainFilterFlags"].strip()
                        print(f"预处理后的trainFilterFlags: {tool_args['trainFilterFlags']}")
                    
                    result = await session.call_tool(tool_name, arguments=tool_args)
                    
                    # 添加工具结果到消息历史
                    # 处理TextContent对象，确保正确提取文本内容
                    content_str = str(result.content)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": content_str
                    })
                
                # 获取下一个回复
                response = self.client.chat.completions.create(
                    model='qwen-plus',
                    max_tokens=2024,
                    tools=self.available_tools,
                    messages=messages
                )
                
                # 如果只有文本回复，则结束处理
                if response.choices[0].message.content and not response.choices[0].message.tool_calls:
                    print(response.choices[0].message.content)
                    process_query = False

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break
                
                await self.process_query(query)
                print("\n")
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Connect to a single MCP server."""
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )

            await session.initialize()
            self.sessions.append(session)

            # List available tools for this session
            response = await session.list_tools()
            tools = response.tools
            print(f"\nConnected to {server_name} with tools:", [t.name for t in tools])

            for tool in tools:
                self.tool_to_session[tool.name] = session
                self.available_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })
        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")

    async def connect_to_servers(self):
        """Connect to all configured MCP servers."""
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)

            servers = data.get("mcpServers", {})

            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise
            
    async def connect_to_server_and_run(self):
        try:
            await self.connect_to_servers()
            await self.chat_loop()
        finally:
            await self.exit_stack.aclose()

async def main():
    chatbot = MCP_ChatBot()
    await chatbot.connect_to_server_and_run()

if __name__ == "__main__":
    asyncio.run(main())
