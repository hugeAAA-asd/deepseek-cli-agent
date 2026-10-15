# DeepSeek 命令行 Agent

一个基于 DeepSeek API 的命令行智能助手，支持工具调用（Function Calling）、多轮对话和真实天气查询。

## 功能

- 🕐 **查询当前时间** — 读本机系统时间，不靠模型编造
- 🧮 **数学计算** — 交给 Python 执行，避免大模型算错
- 🌤️ **查询任意城市实时天气** — 接入和风天气 API，两级请求（城市检索 → 实时天气）
- 💬 **多轮对话** — 保留最近 10 条上下文

## 技术栈

| 组件 | 说明 |
|---|---|
| Python | 3.12.7 |
| DeepSeek API | 大模型能力（`deepseek-flash`） |
| 和风天气 API | 真实天气数据源 |
| requests | HTTP 请求 |
| python-dotenv | 密钥与代码分离 |

## 核心实现

1. **Function Calling 调度** — 定义 3 个工具，模型按用户意图自行选择调用
2. **多工具处理** — 模型一次返回多个 `tool_calls` 时逐个执行并回传结果
3. **两阶段回答** — 第一轮模型决定调工具，第二轮根据工具结果生成自然语言回答
4. **防御性解析** — 所有网络请求带超时；响应经 `safe_json_parse` 校验；第二轮失败自动重试 2 次
5. **密钥隔离** — API Key 通过 `.env` 管理，代码中不硬编码；打包成 exe 后仍从 exe 同目录读取

## 运行方式

### 1. 安装依赖

```powershell
pip install -r requirements.txt
```

### 2. 配置密钥

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`，填入你自己的三个值。

### 3. 运行

```powershell
python main.py
```

## 使用示例

```
你: 西安天气
AI: 西安实时天气:晴,温度18℃,体感18℃,风向东北风,湿度81%

你: 现在几点
AI: 现在是 2026-10-15 14:30:22

你: (10+2)*5
AI: 计算结果是 60
```

## 打包成 exe

```powershell
pyinstaller ds-agent.spec
```

产物在 `dist/ds-agent.exe`。注意 `.env` 需要放在 exe 同目录下（代码里已按 exe 位置查找）。

## 项目结构

```
.
├── main.py              # 主程序：工具定义、Function Calling 调度、对话循环
├── requirements.txt     # 依赖
├── .env.example         # 环境变量模板（不含真实密钥）
├── .gitignore           # 排除密钥、虚拟环境、打包产物
├── ds-agent.spec        # PyInstaller 打包配置
└── README.md
```