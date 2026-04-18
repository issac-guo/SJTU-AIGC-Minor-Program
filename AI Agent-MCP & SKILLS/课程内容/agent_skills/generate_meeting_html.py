#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调用MCP服务器生成会议总结HTML文件
"""

import os
import sys

# 尝试导入MCP客户端
try:
    from mcp import Client
except ImportError:
    try:
        from mcp.client.sync import Client
    except ImportError:
        print("无法导入MCP客户端")
        sys.exit(1)

def generate_html():
    """调用MCP服务器生成HTML文件"""
    try:
        # 创建MCP客户端
        client = Client()
        
        # 调用MCP工具生成HTML文件
        result = client.call("meeting_service.generate_meeting_report", {
            "content_file": "meeting_content.txt",
            "filename": "会议总结.html"
        })
        
        print("调用结果:", result)
        print("HTML文件生成成功！")
        
    except Exception as e:
        print(f"调用失败: {str(e)}")

def main():
    """主函数"""
    # 生成HTML文件
    generate_html()

if __name__ == "__main__":
    main()
