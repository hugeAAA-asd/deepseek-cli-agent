"""RAG 校园知识库问答 - Streamlit 网页版

运行方式：
    streamlit run app.py

功能：
    - 网页聊天界面
    - 对话历史持久化（保存到 chat_history.json，刷新页面不丢）
    - 侧边栏显示知识库统计、清除对话按钮
    - 回答下方可展开查看检索到的参考资料
"""
import json
from pathlib import Path

import streamlit as st

# 页面配置必须在最前面
st.set_page_config(
    page_title="校园知识库问答",
    page_icon="📚",
    layout="wide",
)

from rag_core import search, ask_llm, embed_texts, INDEX_FILE

HISTORY_FILE = Path(__file__).parent / "data" / "chat_history.json"


# ---------------- 初始化（只跑一次） ----------------
@st.cache_resource(show_spinner="正在加载本地 Embedding 模型，约 10 秒...")
def load_model():
    """预热模型，Streamlit 缓存住，之后每次提问不用重新加载。"""
    embed_texts(["预热"])
    return True


def load_history():
    """从 json 文件读取对话历史。"""
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_history(messages):
    """把对话历史存到 json 文件。"""
    # 只存 role + content，sources 太大不存
    to_save = [{"role": m["role"], "content": m["content"]} for m in messages]
    HISTORY_FILE.write_text(
        json.dumps(to_save, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_kb_stats():
    """读 index.json，返回知识库统计。"""
    if not INDEX_FILE.exists():
        return 0, 0
    index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    chunks = index.get("chunks", [])
    total_chars = sum(len(c["text"]) for c in chunks)
    return len(chunks), total_chars


load_model()

# ---------------- 侧边栏 ----------------
with st.sidebar:
    st.header("📊 知识库状态")
    chunk_count, total_chars = get_kb_stats()
    st.metric("文档块数量", chunk_count)
    st.metric("总字符数", f"{total_chars:,}")

    st.divider()
    st.header("⚙️ 设置")

    top_k = st.slider("检索数量 (top_k)", min_value=1, max_value=5, value=3)

    st.divider()
    st.header("🗑️ 对话管理")
    if st.button("清空对话历史", type="secondary"):
        st.session_state.messages = []
        if HISTORY_FILE.exists():
            HISTORY_FILE.unlink()
        st.rerun()

    st.divider()
    st.caption("基于 RAG 检索增强生成\n本地 Embedding + DeepSeek")


# ---------------- 主界面 ----------------
st.title("📚 校园知识库智能问答")
st.caption("输入关于校规、奖学金、宿舍、图书馆等问题，系统会检索学生手册并给出答案")

# 初始化聊天历史（从文件加载）
if "messages" not in st.session_state:
    st.session_state.messages = load_history()

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入框
question = st.chat_input("请输入你的问题，例如：奖学金申请条件是什么？")

if question:
    # 1. 显示用户问题
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 2. 检索 + 回答
    with st.chat_message("assistant"):
        with st.spinner("正在检索知识库并生成回答..."):
            hits = search(question, top_k=top_k)
            contexts = [t for _, t in hits]
            answer = ask_llm(question, contexts)

        st.markdown(answer)

        # 展开参考资料
        with st.expander(f"📄 参考资料（{len(hits)} 段）"):
            for i, (score, text) in enumerate(hits, 1):
                st.markdown(f"**资料 {i}**（相似度 {score:.3f}）")
                st.info(text)
                st.divider()

    # 存历史到 session 和文件
    st.session_state.messages.append({"role": "assistant", "content": answer})
    save_history(st.session_state.messages)
