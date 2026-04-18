import asyncio
import json
import os
from contextlib import AsyncExitStack
import openai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class MCP_ChatBot:
    def __init__(self):
        # 初始化必要的属性
        self.client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        self.exit_stack = AsyncExitStack()
        self.tool_to_session = {}
        self.available_tools = []

    async def process_query(self, query):
        messages = [{'role': 'user', 'content': query}]
        
        while True:
            # 调用OpenAI API获取响应
            response = self.client.chat.completions.create(
                model='qwen-plus',
                max_tokens=2024,
                tools=self.available_tools,
                messages=messages
            )
            
            message = response.choices[0].message
            
            # 处理普通文本回复
            if message.content:
                print(message.content)
                return
            
            # 处理工具调用
            if message.tool_calls:
                messages.append({
                    "role": "assistant", 
                    "content": None,
                    "tool_calls": message.tool_calls
                })
                
                # 执行每个工具调用
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    
                    print(f"调用工具 {tool_name}，参数: {tool_args}")
                    
                    # 调用对应的工具
                    session = self.tool_to_session[tool_name]
                    result = await session.call_tool(tool_name, arguments=tool_args)
                    
                    # 将结果添加到消息历史
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result.content)
                    })
            else:
                break

    async def chat_loop(self):
        """交互式聊天循环"""
        print("\nMCP聊天机器人已启动！")
        print("输入查询内容或'quit'退出。")
        
        while True:
            try:
                query = input("\n查询: ").strip()
                if query.lower() == 'quit':
                    break
                await self.process_query(query)
                print()
            except Exception as e:
                print(f"\n错误: {str(e)}")

    async def load_servers(self):
        """加载并连接到配置的服务器"""
        try:
            # 读取服务器配置
            with open("server_config.json", "r") as file:
                servers = json.load(file).get("mcpServers", {})

            for name, config in servers.items():
                try:
                    # 连接到服务器
                    params = StdioServerParameters(**config)
                    transport = await self.exit_stack.enter_async_context(stdio_client(params))
                    session = await self.exit_stack.enter_async_context(ClientSession(*transport))
                    await session.initialize()

                    # 获取可用工具
                    tools = (await session.list_tools()).tools
                    print(f"\n已连接到 {name}，可用工具: {[t.name for t in tools]}")

                    # 注册工具
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
                    print(f"连接服务器 {name} 失败: {e}")
        except Exception as e:
            print(f"加载服务器配置失败: {e}")
            raise

    async def run(self):
        """启动聊天机器人"""
        try:
            await self.load_servers()
            await self.chat_loop()
        finally:
            await self.exit_stack.aclose()

# 主函数
async def main():
    await MCP_ChatBot().run()

if __name__ == "__main__":
    asyncio.run(main())
