import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# --- Page Configuration ---
st.set_page_config(page_title="University Query System", page_icon="🎓", layout="wide")

# --- Caching Data & Model ---
# Using @st.cache_data and @st.cache_resource ensures the app doesn't reload the dataset 
# or retrain the machine learning model every time the user interacts with the app.

@st.cache_data
def load_data():
    try:
        train_df = pd.read_csv("university_query_train.csv")
        return train_df
    except FileNotFoundError:
        return None

@st.cache_resource
def train_model(train_df):
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z ]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    # Preprocessing
    train_df['Clean_Query'] = train_df['Student_Query'].apply(clean_text)
    X_train = train_df['Clean_Query']
    y_train = train_df['Priority_Label']

    # Model Pipeline
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    model.fit(X_train, y_train)
    
    return model, clean_text

# --- Load and Train ---
train_df = load_data()
model = None
clean_text_func = None

if train_df is not None:
    model, clean_text_func = train_model(train_df)
else:
    st.error("⚠️ Dataset not found. Please ensure 'university_query_train.csv' is in the same directory as this script.")

# --- Helper Functions ---
department_mapping = {
    "fee": "Finance Department",
    "scholarship": "Finance Department",
    "exam": "Examination Cell",
    "hostel": "Hostel Office",
    "wifi": "IT Support",
    "library": "Library",
    "admission": "Admission Cell"
}

def route_department(query):
    query = query.lower()
    for key, value in department_mapping.items():
        if key in query:
            return value
    return "Administration"

def generate_auto_response(query):
    query = query.lower()
    if "scholarship" in query:
        return "Your scholarship-related query has been forwarded to the Finance Department. Expected resolution time: 2-3 working days."
    elif "fee" in query:
        return "Your fee-related query has been registered successfully. Please keep your payment receipt for verification."
    elif "exam" in query:
        return "Your examination query has been forwarded to the Examination Cell. You will receive an update shortly."
    elif "result" in query:
        return "Your result-related query has been registered. The Examination Department will review your request."
    elif "hostel" in query:
        return "Your hostel complaint has been forwarded to the Hostel Administration. Necessary action will be taken soon."
    elif "wifi" in query or "internet" in query:
        return "Your internet/WiFi issue has been sent to the IT Support Team. Please allow some time for troubleshooting."
    elif "library" in query:
        return "Your library-related query has been forwarded to the Library Department."
    elif "admission" in query:
        return "Your admission-related query has been registered successfully."
    elif "placement" in query:
        return "Your placement-related query has been forwarded to the Placement Cell."
    else:
        return "Thank you for contacting the University Helpdesk. Your query has been registered and will be reviewed soon."

def generate_ticket():
    return "TKT" + str(random.randint(1000, 9999))

def summarize_query(query):
    words = query.split()
    return " ".join(words[:8]) + "..."

# --- UI Layout ---
st.title("🎓 University Query Management System")

# Create functional tabs for organization
tab1, tab2 = st.tabs(["💬 Query Helpdesk Chatbot", "📊 Analytics Dashboard"])

# --- TAB 1: Chatbot Interface ---
with tab1:
    st.header("Submit a New Student Query")
    st.write("Enter a student query below. The system will automatically classify its priority, assign it to a department, and generate an automated response.")
    
    user_query = st.text_area("Student Query:", placeholder="E.g., I am unable to submit my examination form")

    if st.button("Submit Query", type="primary"):
        if user_query.strip():
            # Generate Ticket and apply rule-based functions
            ticket_id = generate_ticket()
            department = route_department(user_query)
            response = generate_auto_response(user_query)
            summary = summarize_query(user_query)
            
            # Utilize the trained model to predict Priority Level
            if model is not None:
                cleaned_query = clean_text_func(user_query)
                predicted_priority = model.predict([cleaned_query])[0]
            else:
                predicted_priority = "Unknown (Model missing)"
                
            st.success(f"✅ Query Logged Successfully! Ticket ID: **{ticket_id}**")
            
            # Display results cleanly in columns
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Predicted Priority:** {predicted_priority}")
                st.write(f"**Assigned Department:** {department}")
                st.write(f"**Query Summary:** {summary}")
            with col2:
                st.write("**Auto Response:**")
                st.info(response)
        else:
            st.warning("Please enter a query before submitting.")

# --- TAB 2: Analytics Dashboard ---
with tab2:
    st.header("Historical Query Analytics")
    
    if train_df is not None:
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1:
            # 1. Priority Distribution
            st.subheader("Priority Distribution")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.countplot(x='Priority_Label', data=train_df, ax=ax1, palette="viridis")
            ax1.set_title("Priority Distribution")
            st.pyplot(fig1)
            
            # 2. Resolution Time
            st.subheader("Resolution Time Analysis")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            sns.histplot(train_df['Days_To_Deadline'], bins=20, ax=ax2, color="skyblue")
            ax2.set_title("Resolution Time Analysis")
            st.pyplot(fig2)

        with col_plot2:
            # 3. Department Wise Queries
            st.subheader("Department Wise Queries")
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            sns.countplot(y='Department', data=train_df, ax=ax3, palette="magma")
            ax3.set_title("Department Analytics")
            st.pyplot(fig3)
            
            # 4. Pending Queries (Pie Chart)
            st.subheader("Pending Queries by Priority")
            fig4, ax4 = plt.subplots(figsize=(6, 5))
            train_df['Priority_Label'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax4, colors=['#ff9999','#66b3ff','#99ff99'])
            ax4.set_ylabel('')
            st.pyplot(fig4)