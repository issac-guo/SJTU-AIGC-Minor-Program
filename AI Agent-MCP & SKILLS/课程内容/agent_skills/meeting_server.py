import webbrowser
import os
import sys
import logging
from mcp.server.fastmcp import FastMCP

# 配置日志 - 保持到文件，这是正确的排查方式
logging.basicConfig(
    filename='meeting_server.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# 初始化 FastMCP
# 注意：名字建议与 settings.json 保持一致
mcp = FastMCP("meeting_service")

@mcp.tool()
def generate_meeting_report(content: str = None, filename: str = "会议总结.html", content_file: str = None) -> str:
    """
    将会议记录内容生成美观的 HTML 网页并在浏览器中打开。
    
    Args:
        content: 会议总结的具体内容（支持 HTML 标签或纯文本）。
        filename: 保存的文件名，默认为 '会议总结.html'。
        content_file: 会议总结内容的文件路径，如果提供则从文件中读取内容。
    """
    logger.info(f"正式调用工具：filename={filename}, content_file={content_file}")
    
    # 读取内容
    if content_file and os.path.exists(content_file):
        logger.info(f"从文件读取内容: {content_file}")
        try:
            with open(content_file, 'r', encoding='utf-8') as f:
                content = f.read()
            logger.info(f"成功读取文件内容，长度: {len(content)} 字符")
        except Exception as e:
            logger.error(f"读取文件失败: {str(e)}")
            return f"读取文件失败: {str(e)}"
    elif not content:
        logger.error("未提供内容，也未指定内容文件")
        return "错误：未提供会议总结内容"
    
    # 更加健壮的 HTML 模板
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>会议总结</title>
        <style>
            body {{ font-family: sans-serif; margin: 40px; line-height: 1.6; color: #333; }}
            .container {{ max-width: 800px; margin: auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
            .content {{ background: #f9f9f9; padding: 15px; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>会议总结</h1>
            <div class="content">{content}</div>
        </div>
    </body>
    </html>
    """
    
    # 强制使用绝对路径，避免文件消失在未知位置
    # 我们将其保存在脚本同级目录，或者你可以指定项目目录
    file_path = os.path.abspath(filename)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # 在 Windows 上，webbrowser.open 可能会被阻塞，
        # 但在 MCP 线程中通常没问题
        webbrowser.open(f"file://{file_path}")
        
        logger.info(f"文件已成功写入: {file_path}")
        return f"会议总结已成功生成并打开：{file_path}"
    except Exception as e:
        logger.error(f"写入失败: {str(e)}")
        return f"生成失败: {str(e)}"

if __name__ == "__main__":
    # 重要：不要在这里运行任何 print() 或测试函数！
    # 所有的输出必须通过 logger 写入文件。
    logger.info("MCP Server 正在启动...")
    
    # 检查是否存在meeting_content.txt文件，如果存在则生成HTML文件
    content_file = "meeting_content.txt"
    if os.path.exists(content_file):
        logger.info(f"发现内容文件: {content_file}")
        try:
            result = generate_meeting_report(content_file=content_file, filename="会议总结.html")
            logger.info(f"从文件生成HTML结果: {result}")
        except Exception as e:
            logger.error(f"从文件生成HTML失败: {str(e)}")
    
    try:
        # mcp.run() 会接管 stdout/stdin 进行协议通信
        mcp.run()
    except Exception as e:
        logger.error(f"Server 崩溃: {str(e)}")