---
name: meeting-summary-assistant
description: "Summarizes meeting content and generates an interactive HTML report using the meeting_service MCP tool. Invoke when user needs to summarize discussions or \"upload/sync\" reports."
---

# 会议总结助手 

## 功能
该技能用于总结会议内容，提取关键信息，并通过调用 MCP 工具自动生成美观的 HTML 会议记录网页。
核心要素：
1. 参会人员
2. 议题
3. 决定

## 使用说明
- **分析阶段**：输入会议对话内容，系统自动提取参会人员、议题和决定。
- **总结要求**：每个要素（人员、议题、决定）必须用**一句话**表述，不准分成多条。
- **触发条件**：当用户提到 "上传"、"同步"、"服务器"、"生成网页" 或 "查看记录" 时，必须执行 **自动化报告生成** 流程。

## 财务提醒 (条件触发)
- **触发词**：钱、预算、采购、费用、报销、价格。
- **动作**：必须读取项目中的 `集团财务手册.md`，对比会议决定中的金额。
- **输出**：指出金额是否超标，并明确指出需要的审批人。该内容需包含在最终生成的报告中。

## 自动化报告生成 (MCP 流程)
当触发上传/生成请求时，必须按顺序执行以下操作：

1. **构造内容**：将总结好的“参会人员”、“议题”、“决定”以及“财务提醒（如有）”组合成格式化的 HTML 片段。
2. **调用工具**：**直接使用系统配置的 `meeting_service` MCP 服务**的 `generate_meeting_report` 工具。
   - **参数 content**: 传入格式化后的会议总结内容。
   - **参数 filename**: 默认为 "会议总结.html"。
3. **确认结果**：告知用户 HTML 文件已在项目根目录生成，并已自动在浏览器中打开预览。

## 系统集成
- **MCP 服务**：使用系统已配置的 `meeting_service`，运行在 mcp_exercise 环境中
- **服务配置**：
  ```json
  "meeting_service": {
    "command": "C:\\Users\\ninil\\.conda\\envs\\mcp_exercise\\python.exe",
    "args": [
      "D:\\intelAI4Y\\weizhuanye\\26-03-14\\课程内容\\agent_skills\\meeting_server.py"
    ]
  }
  ```
- **服务调用**：直接通过 MCP 协议调用 `meeting_service.generate_meeting_report` 工具
- **依赖管理**：系统已处理 MCP 服务的依赖，无需本地安装
- **服务状态**：MCP 服务已成功启动并运行，可接收和处理请求

## 输入示例

```
张三：下周二去上海签约。
李四：定外滩附近1200元/晚的酒店，晚上请客户吃3000元的饭。
张三：可以，小李去办吧。
用户：帮我把这个总结上传到服务器。  
```
## 动作示例 (AI 内部逻辑)
1. **生成总结**：
   - 参会人员：张三、李四。
   - 议题：讨论上海签约行程及预算安排。
   - 决定：确定下周二前往上海，预订1200元/晚的酒店并安排3000元的客户晚宴。
2. **财务核查**：经查《集团财务手册.md》，酒店标准超标（标准500元），餐饮超标（标准100元/天），需部门主管审批。
3. **调用 MCP**：`meeting_service.generate_meeting_report(content="...", filename="会议总结.html")`

## 注意事项
- **禁止手动提示**：不要要求用户去终端运行 `python upload.py`，必须由 AI 直接调用 MCP 工具完成。
- **语言**：始终使用中文进行总结和报告生成。
- **验证**：调用工具后，根据工具返回的路径确认操作成功。