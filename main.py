import os
import sys
import json
from errno import EMSGSIZE
from idlelib import history

import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv


if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

load_dotenv(BASE_DIR / ".env")

api_key       = os.getenv("DEEPSEEK_API_KEY")
weather_key   = os.getenv("QWEATHER_KEY")
qweather_host = os.getenv("QWEATHER_HOST", "https://q83jpjtyqr.re.qweatherapi.com")

if not api_key:
    raise SystemExit("错误：.env 里没有 DEEPSEEK_API_KEY")
if not weather_key:
    raise SystemExit("错误：.env 里没有 QWEATHER_KEY")

url = "https://api.deepseek.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

HISTORY_FILE = BASE_DIR / "history.json"

def load_history():
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r",encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[load_history] 历史文件读取失败,已忽略{e}")
        return []
    if not isinstance(data, list):
        print("[load_history]历史文件格式不对,已忽略")
        return []
    return [m for m in data
            if isinstance(m, dict)
            and m.get("role") in ("user", "assistant")
            and not m.get("tool_calls")
            and isinstance(m.get("content"), str)
            and m.get("content").strip()]

def save_history(msgs):
    pure = [m for m in msgs
            if m.get("role") in ("user", "assistant")
            and not m.get("tool_calls")
            and isinstance(m.get("content"), str)
            and m.get("content").strip()]
    tmp = HISTORY_FILE.with_name(HISTORY_FILE.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(pure, f,ensure_ascii=False,indent=2)
        os.replace(tmp, HISTORY_FILE)
    except OSError as e:
        print(f"[save_history]历史保存失败:{e}")

def safe_json_parse(response):
    text = (response.text or "").strip()
    if not text:
        print(f"[safe_json_parse] 响应为空, status={response.status_code}, url={response.url}")
        return None
    if response.status_code != 200:
        print(f"[safe_json_parse] HTTP {response.status_code}, url={response.url}, body={text[:200]!r}")
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"[safe_json_parse] JSON解析失败: {e}, 原文={text[:200]!r}")
        return None


def get_current_time():
    return f"当前时间系统:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


def calculate(expr: str):
    try:
        return f"计算结果:{expr} = {eval(expr)}"
    except Exception as e:
        return f"计算失败,错误:{str(e)}"


def get_weather(city_name):
    def get_city_id(name):
        u = f"{qweather_host}/geo/v2/city/lookup?location={name}&key={weather_key}"
        try:
            res = requests.get(u, timeout=10)
        except Exception as e:
            print(f"[get_city_id] 请求异常: {e}")
            return None
        data = safe_json_parse(res)
        if data is None:
            return None
        if data.get("code") == "200" and data.get("location"):
            return data["location"][0]["id"]
        print(f"[get_city_id] QWeather返回错误码: {data.get('code')}")
        return None

    city_id = get_city_id(city_name)
    if city_id is None:
        return f"找不到[{city_name}]这个城市,请检查名称"

    u = f"{qweather_host}/v7/weather/now?location={city_id}&key={weather_key}"
    try:
        resp = requests.get(u, timeout=10)
    except Exception as e:
        return f"天气请求异常: {e}"

    data = safe_json_parse(resp)
    if data is None:
        return "天气服务返回内容无法解析"

    if data.get("code") == "200":
        now = data["now"]
        return (f"{city_name}实时天气:{now['text']},温度{now['temp']}℃，"
                f"体感{now['feelsLike']}℃，风向{now['windDir']},湿度{now['humidity']}%")
    return f"天气查询失败,错误码{data.get('code')}"


SYSTEM_PROMPT = """你是智能助手，你拥有三个工具：get_current_time、calculate、get_weather。
规则：
1. 用户询问时间、日期，**必须调用 get_current_time**，不要编造时间。
2. 用户需要数学计算，**必须调用 calculate**，数学表达式放到expr参数。
3. 多步任务，先调用工具拿到数据，再继续调用其他工具，不要一步脑补。
4. 用户询问任意城市天气，调用get_weather工具，参数传入城市中文名字。**不要编造天气数据，必须调用工具获取真实结果。**
调用get_weather时，arguments必须输出{"city_name":"城市名"}，不允许arguments为空字符串。
调用 get_weather 时，city_name 只填纯城市名，不要带省份前缀。例如用户说"陕西西安"，你填"西安"。

# 重要！工具调用格式
**你只能使用标准OpenAI tool_calls JSON格式进行工具调用，绝对禁止输出任何DSML、xml标签、<|DSML|>内容。**
当需要调用工具时，返回tool_calls结构，不要直接输出函数调用文本。
不需要工具时，直接输出自然语言回答。
"""

SUMMARY_SYSTEM = "你是一个简洁的中文助手，只用自然语言回答，禁止输出任何工具调用相关内容。"

_history = load_history()
messages = [{"role": "system", "content": SYSTEM_PROMPT}] + _history
if _history:
    print(f"[已加载{len(_history)}条历史对话]")

MAX_MSG_COUNT = 10

tools = [
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "获取本机当前系统时间，当用户询问现在几点、当前日期、此刻时间，调用此函数。",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "calculate",
        "description": "执行数学计算,用户需要算数时调用,传入数学表达式字符串,例如'(10+2)*5'",
        "parameters": {"type": "object",
            "properties": {"expr": {"type": "string", "description": "待计算的数学表达式"}},
            "required": ["expr"]}}},
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "查询指定城市的实时天气,参数city_name是城市中文名称,例如:西安",
        "parameters": {"type": "object",
            "properties": {"city_name": {"type": "string", "description": "城市中文名称"}},
            "required": ["city_name"]}}},
]


print("====Ds简易对话框====")
print("输入exit退出,问时间,天气,计算")

while True:
    user_msg = input("\n你:")
    if user_msg == "exit":
        print("程序结束")
        break

    messages.append({"role": "user", "content": user_msg})
    while len(messages) > MAX_MSG_COUNT:
        messages.pop(1)

    json_body = {
        "model": "deepseek-flash",
        "messages": messages,
        "temperature": 0.7,
        "tools": tools,
        "tool_choice": "auto",
    }

    try:
        resp = requests.post(url, json=json_body, headers=headers, timeout=15)
        # print("\n[第一轮接口返回]:", resp.text)
        resp.raise_for_status()
        resp_data = safe_json_parse(resp)
        if resp_data is None:
            # print("AI:第一轮请求API返回空内容")
            continue

        choice = resp_data["choices"][0]
        finish_reason = choice["finish_reason"]

        if finish_reason == "tool_calls":
            tool_calls_list = choice["message"]["tool_calls"]

            assistant_msg = choice["message"]
            if "reasoning_content" in assistant_msg:
                del assistant_msg["reasoning_content"]
            messages.append(assistant_msg)

            tool_results = []
            for tc in tool_calls_list:
                func_name = tc["function"]["name"]
                arg_str = tc["function"]["arguments"]
                try:
                    args = json.loads(arg_str)
                except json.JSONDecodeError:
                    args = {}

                if func_name == "get_current_time":
                    tool_result = get_current_time()
                elif func_name == "calculate":
                    tool_result = calculate(args.get("expr", ""))
                elif func_name == "get_weather":
                    tool_result = get_weather(args.get("city_name", ""))
                else:
                    tool_result = f"未知工具: {func_name}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result
                })
                # print(f"\n工具执行结果 [{func_name}]: {tool_result}")
                tool_results.append(f"[{func_name}] {tool_result}")

            all_results = "\n".join(tool_results)
            summary_prompt = f"""用户问题：{user_msg}
工具查询结果：
{all_results}
请根据上面的查询结果，用简洁自然中文回答用户。
【严格禁止】不要再调用任何工具，禁止输出任何 tool_calls、DSML、XML、<|...|> 标签。
只输出一段自然语言。"""

            json_body2 = {
                "model": "deepseek-flash",
                "messages": [
                    {"role": "system", "content": SUMMARY_SYSTEM},
                    {"role": "user", "content": summary_prompt},
                ],
                "temperature": 0.3,
            }

            res_data2 = None
            for attempt in range(2):
                resp2 = requests.post(url, json=json_body2, headers=headers, timeout=15)
                # print(f"\n【第二轮 第{attempt+1}次请求返回】:", repr(resp2.text))
                res_data2 = safe_json_parse(resp2)
                if res_data2 is not None:
                    break
                print("本轮返回为空，正在重试...")

            if res_data2 is None:
                print("AI：重试2次依然为空，本次失败，请重新提问")
                continue

            final_ans = res_data2["choices"][0]["message"]["content"]
            print(f"AI: {final_ans}")
            messages.append({"role": "assistant", "content": final_ans})
            save_history(messages)
            continue

        elif finish_reason == "stop":
            ans = choice["message"]["content"]
            print(f"AI: {ans}")
            messages.append({"role": "assistant", "content": ans})
            save_history(messages)

    except requests.exceptions.Timeout:
        print("AI: 请求超时")
    except Exception as e:
        print(f"AI: 异常：{e}")