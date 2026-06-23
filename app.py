"""
University Query Management System - Streamlit App
----------------------------------------------------
Run with:
    streamlit run app.py

Place university_query_train.csv and university_query_test.csv
in the same folder as this file before running.
"""

import re
import random

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    page_title="University Query Management System",
    page_icon="🎓",
    layout="wide",
)

TRAIN_PATH = "university_query_train.csv"
TEST_PATH = "university_query_test.csv"

# ----------------------------------------------------------------------
# TEXT CLEANING
# ----------------------------------------------------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------------------------------------------------
# DEPARTMENT ROUTING & RESPONSES (single source of truth)
# ----------------------------------------------------------------------

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
# DATA + MODEL LOADING (cached so it only runs once per session)
# ----------------------------------------------------------------------

@st.cache_data
def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    train_df["Clean_Query"] = train_df["Student_Query"].apply(clean_text)
    test_df["Clean_Query"] = test_df["Student_Query"].apply(clean_text)
    return train_df, test_df


@st.cache_resource
def train_model(train_df: pd.DataFrame):
    X_train = train_df["Clean_Query"]
    y_train = train_df["Priority_Label"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=42,
        )),
    ])
    model.fit(X_train, y_train)
    return model


def predict_priority(model, query: str) -> str:
    cleaned = clean_text(query)
    return model.predict([cleaned])[0]


# ----------------------------------------------------------------------
# LOAD EVERYTHING (with a friendly error if CSVs are missing)
# ----------------------------------------------------------------------

try:
    train_df, test_df = load_data()
except FileNotFoundError:
    st.error(
        "Could not find `university_query_train.csv` and/or "
        "`university_query_test.csv`. Place both files in the same "
        "folder as `app.py`, then refresh this page."
    )
    st.stop()

model = train_model(train_df)

# ----------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------

st.sidebar.title("🎓 University Helpdesk")
page = st.sidebar.radio(
    "Navigate",
    ["Submit a Query", "Model Performance", "Analytics Dashboard"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Training rows: {len(train_df):,}  \n"
    f"Test rows: {len(test_df):,}"
)

# ----------------------------------------------------------------------
# PAGE 1: SUBMIT A QUERY (the chatbot)
# ----------------------------------------------------------------------

if page == "Submit a Query":
    st.title("🎓 University Query Management System")
    st.write(
        "Type your query below. The system will assign a ticket, "
        "route it to the right department, predict its priority using "
        "a trained machine learning model, and generate an automatic response."
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.form("query_form", clear_on_submit=True):
        query = st.text_area("Describe your issue or question:", height=100)
        submitted = st.form_submit_button("Submit Query")

    if submitted:
        if not query or not query.strip():
            st.warning("Please enter a valid, non-empty query.")
        else:
            ticket_id = generate_ticket()
            department = route_department(query)
            priority = predict_priority(model, query)
            response = generate_auto_response(department)
            summary = summarize_query(query)

            st.session_state.history.insert(0, {
                "ticket_id": ticket_id,
                "query": query,
                "summary": summary,
                "department": department,
                "priority": priority,
                "response": response,
            })

    if st.session_state.history:
        st.markdown("### Ticket Results")
        for item in st.session_state.history:
            priority_color = {
                "High": "🔴",
                "Medium": "🟡",
                "Low": "🟢",
            }.get(item["priority"], "⚪")

            with st.container(border=True):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Ticket ID:** `{item['ticket_id']}`")
                with col2:
                    st.markdown(f"{priority_color} **{item['priority']} priority**")

                st.markdown(f"**Query:** {item['query']}")
                st.markdown(f"**Summary:** {item['summary']}")
                st.markdown(f"**Assigned Department:** {item['department']}")
                st.info(item["response"])
    else:
        st.caption("No queries submitted yet. Try one above.")

# ----------------------------------------------------------------------
# PAGE 2: MODEL PERFORMANCE
# ----------------------------------------------------------------------

elif page == "Model Performance":
    st.title("📊 Model Performance")

    X_test = test_df["Clean_Query"]
    y_test = test_df["Priority_Label"]
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    overlap = set(train_df["Clean_Query"]) & set(test_df["Clean_Query"])
    overlap_pct = len(overlap) / max(len(test_df), 1) * 100

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Test Accuracy", f"{accuracy * 100:.2f}%")
    with col2:
        st.metric("Train/Test Text Overlap", f"{overlap_pct:.1f}%")

    if overlap_pct > 30:
        st.warning(
            "High train/test text overlap detected. The accuracy above is "
            "likely inflated and may not reflect real-world performance on "
            "genuinely new queries."
        )

    st.markdown("### Classification Report")
    report = classification_report(y_test, predictions, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(3))

    st.markdown("### Confusion Matrix")
    labels = sorted(train_df["Priority_Label"].unique())
    cm = confusion_matrix(y_test, predictions, labels=labels)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# ----------------------------------------------------------------------
# PAGE 3: ANALYTICS DASHBOARD
# ----------------------------------------------------------------------

elif page == "Analytics Dashboard":
    st.title("📈 Analytics Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Priority Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        train_df["Priority_Label"].value_counts().plot(
            kind="pie", autopct="%1.1f%%", ax=ax
        )
        ax.set_ylabel("")
        st.pyplot(fig)

    with col2:
        st.markdown("#### Department-Wise Queries")
        fig, ax = plt.subplots(figsize=(6, 4))
        train_df["Department"].value_counts().plot(kind="bar", ax=ax)
        ax.set_ylabel("Number of Queries")
        st.pyplot(fig)

    st.markdown("#### Days to Deadline Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(train_df["Days_To_Deadline"], bins=20, ax=ax)
    st.pyplot(fig)

    st.markdown("#### Most Common Words in Queries")
    from collections import Counter
    words = " ".join(train_df["Clean_Query"])
    top_words = Counter(words.split()).most_common(20)
    word_df = pd.DataFrame(top_words, columns=["Word", "Count"])
    st.dataframe(word_df, use_container_width=True)
