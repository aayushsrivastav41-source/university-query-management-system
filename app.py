import streamlit as st
import pandas as pd
import numpy as np
import faiss
import os
import time
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(
    page_title="UniQuery AI",
    page_icon="🎓",
    layout="wide"
)

# ── SESSION STATE ─────────────────────────────────────────────
if "bg_color" not in st.session_state:
    st.session_state.bg_color = "#0E1117"
if "accent_color" not in st.session_state:
    st.session_state.accent_color = "#00C9A7"
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── THEME MAP ─────────────────────────────────────────────────
themes = {
    "Midnight": {"bg": "#0E1117", "accent": "#00C9A7"},
    "Deep Navy": {"bg": "#0A2342", "accent": "#38BDF8"},
    "Charcoal": {"bg": "#1A1A1A", "accent": "#F59E0B"},
    "Deep Purple": {"bg": "#1A0E2E", "accent": "#A78BFA"},
    "Forest": {"bg": "#0D1F0F", "accent": "#4ADE80"},
}

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <span style="font-size:48px;">🎓</span>
        <h2 style="color:#00C9A7; margin:0; font-size:18px;">UniQuery AI</h2>
        <p style="color:#64748B; font-size:12px; margin:0;">University Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<p style='color:#94A3B8; font-size:12px; font-weight:600; letter-spacing:1px;'>APPEARANCE</p>", unsafe_allow_html=True)
    selected_theme = st.selectbox("Theme", list(themes.keys()), label_visibility="collapsed")
    st.session_state.bg_color = themes[selected_theme]["bg"]
    st.session_state.accent_color = themes[selected_theme]["accent"]

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8; font-size:12px; font-weight:600; letter-spacing:1px;'>API SETTINGS</p>", unsafe_allow_html=True)

    _has_secret_key = False
    try:
        _has_secret_key = bool(st.secrets.get("GROQ_API_KEY", ""))
    except Exception:
        _has_secret_key = False

    if _has_secret_key:
        st.markdown("<p style='color:#4ADE80; font-size:12px;'>✅ API key configured</p>", unsafe_allow_html=True)
    else:
        groq_api_key = st.text_input("Groq API Key", type="password", placeholder="Enter your Groq API key", help="Get your free API key from https://console.groq.com")
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='color:#94A3B8; font-size:12px; font-weight:600; letter-spacing:1px;'>CHAT</p>", unsafe_allow_html=True)

    msg_count = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.markdown(f"<p style='color:#64748B; font-size:13px;'>Messages: {msg_count} / 6</p>", unsafe_allow_html=True)

    if st.button("🗑  Clear History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("<p style='color:#94A3B8; font-size:12px; font-weight:600; letter-spacing:1px;'>QUICK QUERIES</p>", unsafe_allow_html=True)

    quick_queries = [
        "Anti-ragging helpline?",
        "Mess menu Monday?",
        "HOD of CSE?",
        "Library timing?",
        "Placement companies?",
        "Semester 3 CSE subjects?",
    ]
    for q in quick_queries:
        if st.button(q, use_container_width=True, key=f"quick_{q}"):
            st.session_state.quick_query = q
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;">
        <p style="color:#475569; font-size:11px; margin:0;">Built with RAG + Groq LLM</p>
        <p style="color:#475569; font-size:11px; margin:0;">Sentence Transformers + FAISS</p>
        <p style="color:#475569; font-size:11px; margin:4px 0 0 0;">Mohd Adeem Khan</p>
    </div>
    """, unsafe_allow_html=True)

# ── CSS ───────────────────────────────────────────────────────
st.markdown(f"""
<style>
    .stApp {{
        background-color: {st.session_state.bg_color};
    }}
    section[data-testid="stSidebar"] {{
        background-color: #0D0D0D !important;
        border-right: 1px solid #1E293B;
    }}
    .stButton > button {{
        background-color: #1E293B;
        color: #CBD5E1;
        border: 1px solid #334155;
        border-radius: 8px;
        font-size: 12px;
        padding: 6px 12px;
        transition: all 0.2s;
    }}
    .stButton > button:hover {{
        background-color: {st.session_state.accent_color};
        color: #0E1117;
        border-color: {st.session_state.accent_color};
    }}
    .stSelectbox > div > div {{
        background-color: #1E293B;
        color: #FFFFFF;
        border: 1px solid #334155;
        border-radius: 8px;
    }}
    .stChatInput textarea {{
        background-color: #1E293B !important;
        color: #FFFFFF !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
    }}
    .stChatMessage {{
        background-color: transparent !important;
    }}
    [data-testid="stChatMessageContent"] {{
        background-color: #1E293B !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        color: #FFFFFF !important;
    }}
    [data-testid="stChatMessageContent"] p {{
        color: #FFFFFF !important;
    }}
    .stMarkdown p {{
        color: #FFFFFF !important;
    }}
    .stCaption p {{
        color: #64748B !important;
        font-size: 11px !important;
    }}
    .header-container {{
        text-align: center;
        padding: 20px 0 10px 0;
    }}
    .header-title {{
        font-size: 28px;
        font-weight: 700;
        color: {st.session_state.accent_color};
        margin: 0;
    }}
    .header-sub {{
        font-size: 13px;
        color: #64748B;
        margin: 4px 0 0 0;
    }}
    .relevance-bar {{
        height: 4px;
        border-radius: 2px;
        margin-top: 6px;
    }}
    div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stHorizontalBlock"]) {{
        gap: 0px;
    }}
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown(f"""
    <div class="header-container">
        <p class="header-title">🎓 University Query Management System</p>
        <p class="header-sub">Ask anything about hostel, mess, fees, placements, subjects, anti-ragging and more</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr style='border:1px solid #1E293B; margin: 0 0 16px 0;'>", unsafe_allow_html=True)

# ── LOAD RAG ─────────────────────────────────────────────────
@st.cache_resource
def load_embeddings_and_index():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    csv_files = [
        "students.csv", "hostel.csv", "mess.csv", "anti_ragging.csv",
        "academic_info.csv", "exams.csv", "faculty.csv", "fees.csv",
        "admissions.csv", "scholarships.csv", "library.csv", "clubs.csv",
        "transport.csv", "placements.csv", "grievance.csv", "health.csv",
        "rules.csv", "labs.csv", "contacts.csv", "subjects.csv"
    ]

    documents = []
    sources = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            source_name = file.replace(".csv", "")
            for _, row in df.iterrows():
                text = " | ".join([f"{col}: {val}" for col, val in row.items()])
                documents.append(text)
                sources.append(source_name)
        except FileNotFoundError:
            pass

    embeddings = embedding_model.encode(documents)
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return embedding_model, index, documents, sources

def get_client():
    # First check Streamlit Secrets (set in Streamlit Cloud dashboard)
    api_key = ""
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass
    # Fall back to env var or sidebar input if no secret is configured
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")
    if api_key:
        return Groq(api_key=api_key)
    return None

embedding_model, index, documents, sources = load_embeddings_and_index()
client = get_client()

# ── CHECK API KEY ─────────────────────────────────────────────
if client is None:
    st.warning("⚠️ Groq API Key Required", icon="🔑")
    st.info("""
    Please enter your Groq API key in the sidebar to use this assistant.
    
    **How to get a free API key:**
    1. Visit [console.groq.com](https://console.groq.com)
    2. Sign up for a free account
    3. Create an API key
    4. Paste it in the 'Groq API Key' field in the sidebar
    """)
    st.stop()

# ── QUERY FUNCTION ────────────────────────────────────────────
def query_rag(question, chat_history):
    translate_prompt = f"""If the following question is in Hindi, translate it to English.
If it is already in English, return it as is. Return only the question, nothing else.
Question: {question}"""
    translation = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": translate_prompt}]
    )
    search_question = translation.choices[0].message.content.strip()

    question_embedding = embedding_model.encode([search_question])
    question_embedding = np.array(question_embedding).astype("float32")
    distances, indices = index.search(question_embedding, 15)

    retrieved_docs = [documents[i] for i in indices[0]]
    retrieved_sources = [sources[i] for i in indices[0]]
    unique_sources = list(set(retrieved_sources))
    context = "\n".join(retrieved_docs)

    best_distance = float(distances[0][0])
    relevance = max(0, min(100, int(100 - (best_distance * 8))))

    history_text = ""
    recent = chat_history[-12:]
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""You are a helpful university query assistant for an engineering college.
Use ONLY the context provided below to answer the question accurately.
The context contains real university data — trust it completely.

Previous conversation (use this to understand follow-up questions):
{history_text}

Context:
{context}

Current Question: {question}

Answering rules:
- Use the previous conversation to understand follow-up questions and pronouns like it, that, same
- If the user specifies a format like bullet points, numbered points, table or single line — follow it exactly
- If the user specifies a number of points like 3 or 4 — give exactly that many
- If the user asks in Hindi — answer in Hindi
- If no format is specified — give a clear factual answer in 2 to 3 sentences
- Never say information is not available if it exists in the context above
- Always give the actual data values — phone numbers, timings, names etc

End with:
Source: {", ".join(unique_sources)}"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content, relevance

# ── HANDLE QUICK QUERY ────────────────────────────────────────
if "quick_query" in st.session_state and st.session_state.quick_query:
    prompt_text = st.session_state.quick_query
    st.session_state.quick_query = None

    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_text)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Finding answer..."):
            answer, relevance = query_rag(prompt_text, st.session_state.messages)
        placeholder = st.empty()
        displayed = ""
        for word in answer.split(" "):
            displayed += word + " "
            placeholder.markdown(displayed)
            time.sleep(0.03)
        color = "#4ADE80" if relevance >= 70 else "#F59E0B" if relevance >= 40 else "#F87171"
        st.markdown(f"<div class='relevance-bar' style='width:{relevance}%; background:{color};'></div>", unsafe_allow_html=True)
        st.caption(f"Relevance Score: {relevance}%")

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "relevance": relevance
    })

# ── CHAT HISTORY DISPLAY ──────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "🎓" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "relevance" in msg:
            rel = msg["relevance"]
            color = "#4ADE80" if rel >= 70 else "#F59E0B" if rel >= 40 else "#F87171"
            st.markdown(f"<div class='relevance-bar' style='width:{rel}%; background:{color};'></div>", unsafe_allow_html=True)
            st.caption(f"Relevance Score: {rel}%")

# ── CHAT INPUT ────────────────────────────────────────────────
if prompt_text := st.chat_input("Ask anything about your university..."):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_text)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Searching university data..."):
            answer, relevance = query_rag(prompt_text, st.session_state.messages)

        placeholder = st.empty()
        displayed = ""
        for word in answer.split(" "):
            displayed += word + " "
            placeholder.markdown(displayed)
            time.sleep(0.03)

        color = "#4ADE80" if relevance >= 70 else "#F59E0B" if relevance >= 40 else "#F87171"
        st.markdown(f"<div class='relevance-bar' style='width:{relevance}%; background:{color};'></div>", unsafe_allow_html=True)
        st.caption(f"Relevance Score: {relevance}%")

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "relevance": relevance
    })
