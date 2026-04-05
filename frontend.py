import streamlit as st
import requests

API_URL = "http://localhost:8001"

st.set_page_config(
    page_title="PCOS Bot",
    page_icon="🌸",
    layout="centered"
)

# ── Styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .chat-header {
        text-align: center;
        padding: 2rem 1rem 1rem;
        border-bottom: 1px solid #f1f5f9;
        margin-bottom: 1.5rem;
    }
    .chat-header h1 {
        font-size: 2rem;
        background: linear-gradient(135deg, #8a4fff, #ff4f8b);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .chat-header p {
        color: #94a3b8;
        font-size: 0.95rem;
    }
    .source-badge {
        display: inline-block;
        background: #f1f5f9;
        color: #64748b;
        border-radius: 999px;
        padding: 2px 10px;
        font-size: 0.75rem;
        margin-right: 5px;
        margin-top: 4px;
    }
    .disclaimer {
        font-size: 0.78rem;
        color: #94a3b8;
        border-top: 1px dashed #e2e8f0;
        margin-top: 8px;
        padding-top: 6px;
    }
    [data-testid="stChatMessage"] {
        border-radius: 16px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="chat-header">
    <h1>🌸 PCOS Bot</h1>
    <p>Your AI-powered PCOS health companion — evidence-based, empathetic, always available.</p>
</div>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "report_context" not in st.session_state:
    st.session_state.report_context = None

# ── Sidebar — Lab Report (Optional) ───────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 Lab Report (Optional)")
    st.caption("Attach lab values for a more personalised response. This is not required.")

    report_text = st.text_area(
        "Paste biomarker values", height=200,
        placeholder="LH: 12.4 mIU/mL\nFSH: 5.1 mIU/mL\nTestosterone: 85 ng/dL\n..."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Attach", use_container_width=True):
            if report_text.strip():
                st.session_state.report_context = report_text.strip()
                st.success("✅ Report attached")
            else:
                st.warning("Enter some values first.")
    with col2:
        if st.button("Clear", use_container_width=True):
            st.session_state.report_context = None
            st.info("Report removed.")

    if st.session_state.report_context:
        st.success("📎 Lab report attached to your conversation.")

    st.markdown("---")
    st.markdown("### 💡 Try asking:")
    sample_questions = [
        "What are the main symptoms of PCOS?",
        "How does insulin resistance relate to PCOS?",
        "What lifestyle changes help with PCOS?",
        "What is a normal LH/FSH ratio?",
        "Can PCOS cause weight gain?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True, key=q):
            st.session_state.pending_question = q

    st.markdown("---")
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.report_context = None
        st.rerun()

# ── Chat History ──────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🌸" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            unique_src = list(set(s["source"] for s in msg["sources"]))
            badges = "".join(f'<span class="source-badge">{s}</span>' for s in unique_src[:5])
            st.markdown(f'<div style="margin-top:8px">{badges}</div>', unsafe_allow_html=True)
            st.markdown('<div class="disclaimer">⚠️ Not a medical diagnosis. Always consult your doctor.</div>', unsafe_allow_html=True)

# ── Handle sidebar quick-question clicks ──────────────────────────────────
if "pending_question" in st.session_state and st.session_state.pending_question:
    pending = st.session_state.pending_question
    st.session_state.pending_question = None

    st.session_state.messages.append({"role": "user", "content": pending})
    with st.chat_message("user", avatar="👤"):
        st.markdown(pending)

    with st.chat_message("assistant", avatar="🌸"):
        with st.spinner("Searching clinical guidelines..."):
            try:
                response = requests.post(f"{API_URL}/chat", json={
                    "query": pending,
                    "history": [{"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages[:-1]],
                    "report_context": st.session_state.report_context
                }, timeout=90)
                data = response.json()
                answer = data.get("answer", "Sorry, I couldn't get a response.")
                sources = data.get("sources", [])
            except Exception as e:
                answer = f"⚠️ Connection error: {e}"
                sources = []

        st.markdown(answer)
        if sources:
            unique_src = list(set(s["source"] for s in sources))
            badges = "".join(f'<span class="source-badge">{s}</span>' for s in unique_src[:5])
            st.markdown(f'<div style="margin-top:8px">{badges}</div>', unsafe_allow_html=True)
        st.markdown('<div class="disclaimer">⚠️ Not a medical diagnosis. Always consult your doctor.</div>', unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    st.rerun()

# ── Chat Input ────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask anything about PCOS…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🌸"):
        with st.spinner("Searching clinical guidelines..."):
            try:
                response = requests.post(f"{API_URL}/chat", json={
                    "query": prompt,
                    "history": [{"role": m["role"], "content": m["content"]}
                                for m in st.session_state.messages[:-1]],
                    "report_context": st.session_state.report_context
                }, timeout=90)
                data = response.json()
                answer = data.get("answer", "Sorry, I couldn't get a response.")
                sources = data.get("sources", [])
            except Exception as e:
                answer = f"⚠️ Could not connect to the API at {API_URL}. Make sure the server is running using `python3 run.py`."
                sources = []

        st.markdown(answer)
        if sources:
            unique_src = list(set(s["source"] for s in sources))
            badges = "".join(f'<span class="source-badge">{s}</span>' for s in unique_src[:5])
            st.markdown(f'<div style="margin-top:8px">{badges}</div>', unsafe_allow_html=True)
        st.markdown('<div class="disclaimer">⚠️ Not a medical diagnosis. Always consult your doctor.</div>', unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
