"""RAG 核心工具：切块、向量化、相似度、检索、调用 DeepSeek。

支持两种向量化方式，在 .env 里用 EMBED_PROVIDER 切换：
  local —— 本地模型（默认），不需要任何 key
  api   —— 远程 embedding 服务，需要 EMBED_API_KEY

这个文件是四个练习步骤共用的"零件库"，本身不单独运行。
"""
import json
import math
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

# ---------------- 配置（全部从 .env 读，代码里不写死任何 key）----------------
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "local")

# 本地模型
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
# BGE 中文模型的官方建议：查询语句加此前缀，检索效果更好（只用于 query，不用于文档）
LOCAL_QUERY_INSTRUCTION = os.getenv(
    "LOCAL_QUERY_INSTRUCTION", "为这个句子生成表示以用于检索相关文章：")

# 远程 embedding 服务
EMBED_URL = os.getenv("EMBED_URL", "https://api.siliconflow.cn/v1/embeddings")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "")

# 对话模型
CHAT_URL = os.getenv("DEEPSEEK_CHAT_URL", "https://api.deepseek.com/v1/chat/completions")
CHAT_MODEL = os.getenv("CHAT_MODEL", "deepseek-flash")
# 思考模式：disabled = 直接回答（快）；enabled = 先输出推理链（慢，文档问答不需要）
CHAT_THINKING = os.getenv("CHAT_THINKING", "disabled")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

INDEX_FILE = BASE_DIR / "data" / "index.json"
# 语料文件，可在 .env 里用 CORPUS_FILE 指定
CORPUS_FILE = BASE_DIR / "data" / os.getenv("CORPUS_FILE", "corpus_big.txt")

# 检索默认返回几块。改 .env 里的 TOP_K 即可，三个 step 文件都会跟着变。
TOP_K = int(os.getenv("TOP_K", "3"))

_LOCAL_MODEL = None


def require(name, value):
    if not value:
        raise SystemExit(f"错误：.env 里缺少 {name}，请参照 .env.example 填写")


# ---------------- 1. 切块 ----------------
def split_text(text, chunk_size=400, overlap=80):
    """把长文本切成带重叠的块，尽量不把一句话从中间切断。

    chunk_size: 每块的目标字符数
    overlap:    相邻两块重叠的字符数（防止答案正好卡在切口处丢失）
    """
    overlap = max(0, min(overlap, chunk_size // 4))
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    # 在中文句末标点和换行之后断开，标点保留在前一句
    parts = re.split(r"(?<=[。！？!?；;\n])", text)

    chunks, buf = [], ""
    for p in parts:
        if not p.strip():
            continue
        if len(buf) + len(p) > chunk_size and buf:
            chunks.append(buf)
            buf = buf[-overlap:] if overlap else ""
        buf += p
    if buf.strip():
        chunks.append(buf)
    return chunks


# ---------------- 2. 向量化 ----------------
def _local_model_dir():
    """在本地缓存里找到模型快照目录。

    给 SentenceTransformer 传「目录路径」而不是「模型名」时，它完全不会联网。
    这比设置 HF_HUB_OFFLINE 更彻底：transformers 5.x 加载时仍会去
    huggingface.co 探测 adapter_config.json，且带 5 次指数退避重试
    （1+2+4+8+16 秒），国内会卡几十秒到几分钟，表现就是"加载不出来"。
    """
    home = os.getenv("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    snaps = Path(home) / "hub" / f"models--{LOCAL_MODEL_NAME.replace('/', '--')}" / "snapshots"
    if not snaps.is_dir():
        return None
    found = [
        d for d in snaps.iterdir()
        if d.is_dir() and ((d / "model.safetensors").exists() or (d / "pytorch_model.bin").exists())
    ]
    if not found:
        return None
    return max(found, key=lambda d: d.stat().st_mtime)

def search_knowledge_base(query:str) -> str:
    """搜索校园知识库,返回相关文档内容"""
    hits = search(query, top_k=3)
    if not hits:
        return "知识库中没有找到相关内容."
    result = ""
    for i,(score,text) in enumerate(hits,1):
        result += f"【资料{i}】(相似度{score:.3f})\n {text}\n\n"
    return result

def _embed_local(texts, is_query=False):
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        # 关键：强制离线加载。
        # 模型已在本地缓存时，sentence-transformers 仍会去 huggingface.co
        # 查询最新版本；国内访问不了，会卡几十秒甚至几分钟（看起来像"加载不出来"）。
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        from sentence_transformers import SentenceTransformer

        print(f"  正在加载本地模型 {LOCAL_MODEL_NAME} ...")
        target = _local_model_dir()
        if target is not None:
            print(f"  命中本地缓存，离线加载（不联网）")
            target = str(target)
        else:
            print("  本地缓存未命中，将尝试联网下载（需要 HF_ENDPOINT 镜像）")
            target = LOCAL_MODEL_NAME
        try:
            _LOCAL_MODEL = SentenceTransformer(target)
        except Exception as e:
            raise SystemExit(
                f"加载本地模型失败：{e}\n"
                f"\n如果模型还没下载过，请先在当前终端执行：\n"
                f'    $env:HF_ENDPOINT = "https://hf-mirror.com"\n'
                f"然后重新运行本程序（首次会下载约 91 MB）。"
            )

    if is_query and LOCAL_QUERY_INSTRUCTION:
        texts = [LOCAL_QUERY_INSTRUCTION + t for t in texts]

    vectors = _LOCAL_MODEL.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def _embed_api(texts, batch_size=10):
    require("EMBED_API_KEY", EMBED_API_KEY)
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            resp = requests.post(
                EMBED_URL,
                headers={"Authorization": f"Bearer {EMBED_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": EMBED_MODEL, "input": batch},
                timeout=60,
            )
        except requests.RequestException as e:
            raise SystemExit(f"Embedding 请求异常：{e}")
        if resp.status_code != 200:
            raise SystemExit(f"Embedding 失败 {resp.status_code}：{resp.text[:300]}")
        data = resp.json()["data"]
        # 关键：服务端不保证返回顺序，按 index 排序后才和输入对得上
        data.sort(key=lambda d: d["index"])
        vectors.extend(d["embedding"] for d in data)
        print(f"  向量化进度 {min(i + batch_size, len(texts))}/{len(texts)}")
    return vectors


def embed_texts(texts, is_query=False):
    """把一批文字变成向量，返回顺序与输入严格一致。

    is_query=True 表示这是"检索用的问句"（本地模型会给它加检索指令前缀）。
    """
    if EMBED_PROVIDER == "local":
        return _embed_local(texts, is_query=is_query)
    return _embed_api(texts)


# ---------------- 3. 余弦相似度 ----------------
def cosine(a, b):
    """返回 -1~1，越接近 1 越相似。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------- 4. 检索 ----------------
def search(question, top_k=None):
    """返回 [(相似度, 原文块), ...]，按相似度从高到低。

    top_k 不传时读 .env 里的 TOP_K。
    """
    if top_k is None:
        top_k = TOP_K
    """返回 [(相似度, 原文块), ...]，按相似度从高到低。"""
    if not INDEX_FILE.exists():
        raise SystemExit("还没有索引文件，请先运行 step2_build_index.py")
    index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    q_vec = embed_texts([question], is_query=True)[0]
    scored = [(cosine(q_vec, c["embedding"]), c["text"]) for c in index["chunks"]]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# ---------------- 5. 调 DeepSeek 生成回答 ----------------
SYSTEM_PROMPT = (
    "你是一个严谨的文档问答助手。严格只根据用户提供的【资料】回答问题。"
    "如果资料里没有相关信息，就直接回答：资料中没有提到相关内容。"
    "不要使用资料以外的知识，不要编造。"
)


def ask_llm(question, contexts):
    require("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY)
    context_text = "\n\n".join(f"【资料{i + 1}】\n{t}" for i, t in enumerate(contexts))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context_text}\n\n【问题】\n{question}"},
    ]
    try:
        resp = requests.post(
            CHAT_URL,
            headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                     "Content-Type": "application/json"},
            json={
                "model": CHAT_MODEL,
                "messages": messages,
                "temperature": 0.2,
                "thinking": {"type":"disabled"},
            },
            timeout=60,
        )
    except requests.RequestException as e:
        raise SystemExit(f"DeepSeek 请求异常：{e}")
    if resp.status_code != 200:
        raise SystemExit(f"DeepSeek 失败 {resp.status_code}：{resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"]