#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件上传脚本
用于上传文件到指定网页
"""

import os
import sys
import argparse
import webbrowser

def create_html_summary(content, output_file):
    """
    创建HTML格式的会议总结
    
    Args:
        content: 会议总结内容
        output_file: 输出HTML文件路径
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>会议总结</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
                border-bottom: 2px solid #4CAF50;
                padding-bottom: 10px;
            }
            .summary-item {
                margin: 20px 0;
                padding: 15px;
                background-color: #f9f9f9;
                border-left: 4px solid #4CAF50;
                border-radius: 5px;
            }
            .label {
                font-weight: bold;
                color: #4CAF50;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>会议总结</h1>
            {content}
        </div>
    </body>
    </html>
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"HTML文件已创建: {output_file}")
    webbrowser.open(output_file)
    return output_file

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='会议总结上传脚本')
    parser.add_argument('--content', '-c', help='会议总结内容')
    parser.add_argument('--output', '-o', default='会议总结.html', help='输出HTML文件路径')
    
    args = parser.parse_args()
    
    if not args.content:
        # 默认会议总结内容
        content = """
        <div class="summary-item">
            <span class="label">参会人员：</span>老陈、小李
        </div>
        <div class="summary-item">
            <span class="label">议题：</span>讨论去上海与大客户签约的出差安排事宜。
        </div>
        <div class="summary-item">
            <span class="label">决定：</span>下周二去上海签约，预订外滩附近1200元/晚的酒店，晚上请客户吃3000元左右的饭（6人，人均500元），小李尽快提交出差申请以便审批后订票。
        </div>
        <div class="summary-item">
            <span class="label">财务提醒：</span>住宿费用1200元/晚超过集团财务手册一线城市住宿标准（不超过500元/晚），餐饮费用3000元超过国内出差餐饮补贴标准（100元/天），两项支出合计4200元，需经部门主管审批。
        </div>
        """
    else:
        content = args.content
    
    create_html_summary(content, args.output)

if __name__ == '__main__':
    main()
