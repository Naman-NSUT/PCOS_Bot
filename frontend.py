import streamlit as st
import requests

API_URL = "http://127.0.0.1:8003"

st.set_page_config(
    page_title="Maya — PCOS Health Assistant",
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
    .phase-badge {
        display: inline-block;
        background: linear-gradient(135deg, #8a4fff22, #ff4f8b22);
        color: #8a4fff;
        border-radius: 999px;
        padding: 2px 12px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    [data-testid="stChatMessage"] {
        border-radius: 16px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="chat-header">
    <h1>🌸 Maya</h1>
    <p>Your AI health consultation companion — empathetic, evidence-based, always here for you.</p>
</div>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None

# Durable identity. st.session_state alone is NOT enough — it is per browser tab
# and dies on refresh, so the user would look brand new every reload. The token
# is mirrored into st.query_params, which survives a refresh and can be
# bookmarked to resume a consultation history.
if "user_token" not in st.session_state:
    st.session_state.user_token = st.query_params.get("u")

if "is_returning" not in st.session_state:
    st.session_state.is_returning = False


def _remember_token(token):
    """Persist the identity token so a refresh still recognises this user."""
    if token and token != st.session_state.user_token:
        st.session_state.user_token = token
    if token:
        st.query_params["u"] = token

if "phase" not in st.session_state:
    st.session_state.phase = 1

if "report_context" not in st.session_state:
    st.session_state.report_context = None

if "initialized" not in st.session_state:
    st.session_state.initialized = False

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌸 Maya")
    st.caption("AI-powered PCOS health consultation")

    # Phase indicator
    phase_names = {1: "First Contact", 2: "Intake", 3: "Symptom Exploration", 4: "Guidance"}
    current_phase = st.session_state.phase
    st.markdown(f"**Current phase:** {phase_names.get(current_phase, 'Unknown')}")

    st.markdown("---")

    # Lab Report (only visible in Phase 3+)
    if current_phase >= 3:
        st.markdown("### 🧬 Lab Report (Optional)")
        st.caption("Attach lab values for more personalised guidance.")
        report_text = st.text_area(
            "Paste biomarker values", height=150,
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
            if st.button("Clear Report", use_container_width=True):
                st.session_state.report_context = None
                st.info("Report removed.")

        if st.session_state.report_context:
            st.success("📎 Lab report attached.")

    st.markdown("---")

    if st.session_state.is_returning:
        st.caption("🧠 Maya remembers your previous consultations.")
    elif st.session_state.user_token:
        st.caption("🧠 This consultation will be remembered next time.")

    # Starts a fresh consultation but KEEPS identity, so memory carries over.
    if st.button("🗑 New Consultation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.session_state.phase = 1
        st.session_state.report_context = None
        st.session_state.initialized = False
        st.rerun()

    with st.expander("Privacy"):
        st.caption(
            "Your consultations are stored locally on the server so Maya can "
            "remember you. You can erase everything at any time."
        )
        if st.button("Forget me and delete my data", use_container_width=True):
            try:
                r = requests.delete(
                    f"{API_URL}/chat/memory",
                    json={"user_token": st.session_state.user_token},
                    timeout=60,
                )
                if r.ok:
                    st.session_state.clear()
                    st.query_params.clear()
                    st.success("All your data was deleted.")
                    st.rerun()
                else:
                    st.warning("Nothing to delete.")
            except Exception:
                st.warning("Could not reach the server.")


# ── Initialize: Get Maya's greeting ──────────────────────────────────────
def _initialize_session():
    """Call /chat/new to get Maya's greeting and start the session."""
    try:
        response = requests.post(
            f"{API_URL}/chat/new",
            json={"user_token": st.session_state.user_token},
            timeout=300,
        )
        data = response.json()
        st.session_state.session_id = data.get("session_id")
        st.session_state.phase = data.get("phase", 1)
        st.session_state.is_returning = data.get("is_returning_user", False)
        _remember_token(data.get("user_token"))
        greeting = data.get("answer", "Hi there, I'm Maya. What's your name?")
        st.session_state.messages.append({
            "role": "assistant",
            "content": greeting,
            "sources": [],
        })
        st.session_state.initialized = True
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ Could not connect to Maya at {API_URL}. Make sure the server is running with `python3 run.py`.",
            "sources": [],
        })
        st.session_state.initialized = True


if not st.session_state.initialized:
    _initialize_session()

# ── Chat History ──────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "🌸" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            unique_src = list(set(s["source"] for s in msg["sources"]))
            badges = "".join(f'<span class="source-badge">{s}</span>' for s in unique_src[:5])
            st.markdown(f'<div style="margin-top:8px">{badges}</div>', unsafe_allow_html=True)

# ── Chat Input ────────────────────────────────────────────────────────────
if prompt := st.chat_input("Type your message…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🌸"):
        with st.spinner("Maya is thinking..."):
            try:
                response = requests.post(f"{API_URL}/chat", json={
                    "query": prompt,
                    "session_id": st.session_state.session_id,
                    "user_token": st.session_state.user_token,
                }, timeout=600)
                data = response.json()
                _remember_token(data.get("user_token"))
                answer = data.get("answer", "I'm sorry, I couldn't process that.")
                sources = data.get("sources", [])
                st.session_state.phase = data.get("phase", st.session_state.phase)
                st.session_state.session_id = data.get("session_id", st.session_state.session_id)
            except Exception as e:
                answer = f"⚠️ Could not connect to Maya at {API_URL}. Make sure the server is running with `python3 run.py`."
                sources = []

        st.markdown(answer)
        if sources:
            unique_src = list(set(s["source"] for s in sources))
            badges = "".join(f'<span class="source-badge">{s}</span>' for s in unique_src[:5])
            st.markdown(f'<div style="margin-top:8px">{badges}</div>', unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    st.rerun()
