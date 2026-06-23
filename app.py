"""
University Query Management System - Streamlit App (Chat-Style)
-------------------------------------------------------------------
Run with:
    streamlit run app.py

Place these files in the same folder as this app.py before running:
    university_query_train.csv
    university_query_test.csv
    university_course_facts.csv
"""

import re
import random
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------

st.set_page_config(
    page_title="University Helpdesk",
    page_icon="🎓",
    layout="centered",
)

TRAIN_PATH = "university_query_train.csv"
TEST_PATH = "university_query_test.csv"
FACTS_PATH = "university_course_facts.csv"

# ----------------------------------------------------------------------
# LIGHT / MINIMAL THEME
# ----------------------------------------------------------------------

PRIMARY = "#2563EB"      # clean blue accent
BG = "#FAFAFA"
CARD_BG = "#FFFFFF"
BORDER = "#E5E7EB"
TEXT_DARK = "#111827"
TEXT_MUTED = "#6B7280"
SUCCESS_BG = "#ECFDF5"
SUCCESS_TEXT = "#047857"
WARN_BG = "#FFFBEB"
WARN_TEXT = "#92400E"

st.markdown(f"""
<style>
    .stApp {{
        background-color: {BG};
    }}
    .block-container {{
        padding-top: 2rem;
        max-width: 720px;
    }}
    /* Chat bubbles */
    .chat-bubble-user {{
        background-color: {PRIMARY};
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 14px 14px 2px 14px;
        margin: 0.4rem 0;
        max-width: 85%;
        margin-left: auto;
        font-size: 0.95rem;
        line-height: 1.4;
    }}
    .chat-bubble-bot {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        color: {TEXT_DARK};
        padding: 0.85rem 1.1rem;
        border-radius: 14px 14px 14px 2px;
        margin: 0.4rem 0;
        max-width: 92%;
        font-size: 0.95rem;
        line-height: 1.5;
    }}
    .fact-pill {{
        display: inline-block;
        background-color: {SUCCESS_BG};
        color: {SUCCESS_TEXT};
        padding: 0.5rem 0.9rem;
        border-radius: 10px;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 0.95rem;
    }}
    .fact-pill-warn {{
        display: inline-block;
        background-color: {WARN_BG};
        color: {WARN_TEXT};
        padding: 0.5rem 0.9rem;
        border-radius: 10px;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }}
    .meta-row {{
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 0.6rem;
        font-size: 0.78rem;
    }}
    .meta-chip {{
        background-color: #F3F4F6;
        color: {TEXT_MUTED};
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        border: 1px solid {BORDER};
    }}
    .priority-high {{ background-color: #FEF2F2; color: #B91C1C; border-color: #FECACA; }}
    .priority-medium {{ background-color: #FFFBEB; color: #92400E; border-color: #FDE68A; }}
    .priority-low {{ background-color: #ECFDF5; color: #047857; border-color: #A7F3D0; }}

    .app-header {{
        text-align: center;
        margin-bottom: 1.5rem;
    }}
    .app-header h1 {{
        font-size: 1.6rem;
        color: {TEXT_DARK};
        margin-bottom: 0.2rem;
    }}
    .app-header p {{
        color: {TEXT_MUTED};
        font-size: 0.9rem;
    }}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# SHARED LOGIC (identical to the verified version — unchanged)
# ----------------------------------------------------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


DEPARTMENT_RULES = [
    (r"\bhall ticket\b", "Examination Cell"),
    (r"\badmit card\b", "Examination Cell"),
    (r"\bexam(ination)?\b", "Examination Cell"),
    (r"\bresult\b", "Examination Cell"),
    (r"\bscholarship\b", "Finance Office"),
    (r"\bfee\b|\bpayment\b", "Finance Office"),
    (r"\bhostel\b", "Hostel Office"),
    (r"\bwifi\b|\binternet\b", "IT Support"),
    (r"\bportal\b|\bpassword\b", "IT Support"),
    (r"\blibrary\b", "Library"),
    (r"\badmission\b", "Admission Cell"),
    (r"\bplacement\b|\binternship\b", "Placement Cell"),
]

RESPONSE_TEMPLATES = {
    "Examination Cell": "Your examination-related query has been forwarded to the Examination Cell. You will receive an update shortly.",
    "Finance Office": "Your fee/scholarship-related query has been registered. Please keep your payment receipt for verification.",
    "Hostel Office": "Your hostel-related complaint has been forwarded to the Hostel Office. Necessary action will be taken soon.",
    "IT Support": "Your portal/internet issue has been sent to the IT Support team. Please allow some time for troubleshooting.",
    "Library": "Your library-related query has been forwarded to the Library Department.",
    "Admission Cell": "Your admission-related query has been registered successfully.",
    "Placement Cell": "Your placement/internship-related query has been forwarded to the Placement Cell.",
    "Administration": "Thank you for contacting the University Helpdesk. Your query has been registered and will be reviewed soon.",
}


def route_department(query: str) -> str:
    q = query.lower()
    for pattern, dept in DEPARTMENT_RULES:
        if re.search(pattern, q):
            return dept
    return "Administration"


def generate_auto_response(department: str) -> str:
    return RESPONSE_TEMPLATES.get(department, RESPONSE_TEMPLATES["Administration"])


def generate_ticket() -> str:
    return "TKT" + str(random.randint(1000, 9999))


def summarize_query(query: str, max_words: int = 8) -> str:
    words = query.split()
    if len(words) <= max_words:
        return query
    return " ".join(words[:max_words]) + "..."


# ----------------------------------------------------------------------
# FACTUAL ANSWER LOOKUP MODULE (identical to the verified version)
# ----------------------------------------------------------------------

COURSE_ALIASES = {
    "BTECH_CSE": ["btech cse", "b.tech cse", "btech computer science", "cse", "b.tech computer science engineering"],
    "BTECH_ECE": ["btech ece", "b.tech ece", "electronics", "ece"],
    "BTECH_MECH": ["btech mech", "mechanical", "mech"],
    "BTECH_CIVIL": ["btech civil", "civil engineering", "civil"],
    "BCA": ["bca", "bachelor of computer applications"],
    "MCA": ["mca", "master of computer applications"],
    "MBA": ["mba", "master of business administration"],
    "BBA": ["bba", "bachelor of business administration"],
    "BSC_AGRI": ["bsc agriculture", "b.sc agriculture", "agriculture"],
    "MSC_CS": ["msc cs", "m.sc computer science", "msc computer science"],
    "BCOM": ["bcom", "b.com", "bachelor of commerce"],
    "BA_ENGLISH": ["ba english", "b.a english", "bachelor of arts"],
    "DIPLOMA_CSE": ["diploma cse", "diploma computer science", "diploma in computer science engineering"],
    "LLB": ["llb", "bachelor of laws", "law"],
    "BPHARMA": ["bpharma", "b.pharma", "bachelor of pharmacy", "pharmacy"],
}

FACT_INTENTS = [
    (r"\bhostel fee|hostel charge|hostel cost\b", "Hostel_Fee_INR", "Hostel Fee", "currency"),
    (r"\blibrary fine\b", "Library_Fine_Per_Day_INR", "Library Fine (per day)", "currency"),
    (r"\bscholarship amount|scholarship money|how much.*scholarship\b", "Scholarship_Amount_INR", "Scholarship Amount", "currency"),
    (r"\bscholarship deadline|scholarship last date|when.*scholarship\b", "Scholarship_Deadline", "Scholarship Deadline", "date"),
    (r"\badmission fee\b", "Admission_Fee_INR", "Admission Fee", "currency"),
    (r"\badmission deadline|admission last date|when.*admission\b", "Admission_Deadline", "Admission Deadline", "date"),
    (r"\bexam form deadline|exam form last date\b", "Exam_Form_Deadline", "Exam Form Submission Deadline", "date"),
    (r"\bexam (start )?date|when.*exam|exam timetable|exam schedule\b", "Exam_Start_Date", "Exam Start Date", "date"),
    (r"\bplacement (registration )?deadline|placement last date\b", "Placement_Registration_Deadline", "Placement Registration Deadline", "date"),
    (r"\bconvocation\b", "Convocation_Date", "Convocation Date", "date"),
    (r"\b(college |tuition |annual |course )?fee\b", "Annual_Fee_INR", "Annual Course Fee", "currency"),
]


def resolve_course(course_input: str, facts_df: pd.DataFrame):
    if not course_input:
        return None
    q = course_input.strip().lower()
    q_norm = re.sub(r"[^a-z0-9 ]", "", q)

    upper = course_input.strip().upper().replace(" ", "_")
    if upper in facts_df["Course"].values:
        return upper

    best_match = None
    best_len = 0
    for code, aliases in COURSE_ALIASES.items():
        for alias in aliases:
            if alias in q_norm and len(alias) > best_len:
                best_match = code
                best_len = len(alias)
    return best_match


def detect_fact_intent(query: str):
    q = query.lower()
    for pattern, column, label, fmt in FACT_INTENTS:
        if re.search(pattern, q):
            return column, label, fmt
    return None


def format_value(value, fmt):
    if fmt == "currency":
        return f"Rs. {int(value):,}"
    return str(value)


def answer_factual_query(query: str, course_input: str, facts_df: pd.DataFrame):
    intent = detect_fact_intent(query)
    if intent is None:
        return None

    column, label, fmt = intent
    course_code = resolve_course(course_input, facts_df)

    if course_code is None:
        return {
            "matched": True,
            "course_resolved": False,
            "message": (
                "I can look that up, but I need to know your course/branch "
                "first. Please select your course from the sidebar and ask again."
            ),
        }

    row = facts_df[facts_df["Course"] == course_code].iloc[0]
    value = row[column]
    course_name = row["Course_Full_Name"]

    return {
        "matched": True,
        "course_resolved": True,
        "course_code": course_code,
        "course_name": course_name,
        "label": label,
        "value": value,
        "formatted_value": format_value(value, fmt),
        "message": f"Your {label} ({course_name}) is {format_value(value, fmt)}.",
    }


def predict_priority(model, query: str) -> str:
    cleaned = clean_text(query)
    return model.predict([cleaned])[0]


# ----------------------------------------------------------------------
# DATA + MODEL LOADING (cached)
# ----------------------------------------------------------------------

@st.cache_data
def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    train_df["Clean_Query"] = train_df["Student_Query"].apply(clean_text)
    test_df["Clean_Query"] = test_df["Student_Query"].apply(clean_text)

    facts_df = pd.read_csv(FACTS_PATH)
    facts_df["Course"] = facts_df["Course"].str.upper()

    return train_df, test_df, facts_df


@st.cache_resource
def train_model(train_df: pd.DataFrame):
    X_train = train_df["Clean_Query"]
    y_train = train_df["Priority_Label"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("classifier", RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)),
    ])
    model.fit(X_train, y_train)
    return model


try:
    train_df, test_df, facts_df = load_data()
except FileNotFoundError:
    st.error(
        "Could not find one of the required CSV files: "
        "`university_query_train.csv`, `university_query_test.csv`, or "
        "`university_course_facts.csv`. Place all three in the same "
        "folder as `app.py`, then refresh this page."
    )
    st.stop()

model = train_model(train_df)

# ----------------------------------------------------------------------
# SIDEBAR — course/student context + quick stats
# ----------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Your Details")
    course_options = ["-- Select your course --"] + sorted(facts_df["Course_Full_Name"].tolist())
    course_choice = st.selectbox("Course / Branch", course_options)
    student_id = st.text_input("Student ID (optional)")

    st.markdown("---")
    st.markdown("### System Info")
    st.caption(f"Training data: {len(train_df):,} queries")
    st.caption(f"Courses covered: {len(facts_df)}")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ----------------------------------------------------------------------
# MAIN CHAT PAGE
# ----------------------------------------------------------------------

st.markdown("""
<div class="app-header">
    <h1>🎓 University Helpdesk</h1>
    <p>Ask about fees, deadlines, or any issue — get an instant answer, ticket, and routing.</p>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render conversation history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-bubble-user">{msg["text"]}</div>', unsafe_allow_html=True)
    else:
        bubble_html = '<div class="chat-bubble-bot">'

        fact_result = msg.get("fact_result")
        if fact_result is not None:
            if fact_result["course_resolved"]:
                bubble_html += f'<div class="fact-pill">✅ {fact_result["message"]}</div><br>'
            else:
                bubble_html += f'<div class="fact-pill-warn">⚠️ {fact_result["message"]}</div><br>'

        bubble_html += msg["text"]

        priority_class = {
            "High": "priority-high",
            "Medium": "priority-medium",
            "Low": "priority-low",
        }.get(msg.get("priority"), "")

        bubble_html += (
            '<div class="meta-row">'
            f'<span class="meta-chip">🎫 {msg.get("ticket_id", "")}</span>'
            f'<span class="meta-chip">🏢 {msg.get("department", "")}</span>'
            f'<span class="meta-chip {priority_class}">⚡ {msg.get("priority", "")} priority</span>'
            '</div>'
        )
        bubble_html += "</div>"
        st.markdown(bubble_html, unsafe_allow_html=True)

# Chat input
user_query = st.chat_input("Type your question here...")

if user_query and user_query.strip():
    course_input = "" if course_choice.startswith("--") else course_choice

    # Step 1: factual lookup
    fact_result = answer_factual_query(user_query, course_input, facts_df)

    # Step 2: ticket + routing + priority + response (always runs)
    ticket_id = generate_ticket()
    department = route_department(user_query)
    priority = predict_priority(model, user_query)
    response = generate_auto_response(department)

    st.session_state.messages.append({"role": "user", "text": user_query})
    st.session_state.messages.append({
        "role": "bot",
        "text": response,
        "ticket_id": ticket_id,
        "department": department,
        "priority": priority,
        "fact_result": fact_result,
    })
    st.rerun()

if not st.session_state.messages:
    st.markdown(
        '<p style="text-align:center; color:#9CA3AF; font-size:0.85rem; margin-top:2rem;">'
        "Try asking: \"What is my college fee?\" or \"When is the exam form deadline?\""
        "</p>",
        unsafe_allow_html=True,
    )
