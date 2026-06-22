import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Title

st.title("University Query Management System")

st.write("This system predicts the priority level of student queries.")

# Load datasets

train_data = pd.read_csv("university_query_train.csv")
test_data = pd.read_csv("university_query_test.csv")

# Features and labels

X_train = train_data["Student_Query"]
y_train = train_data["Priority_Label"]

X_test = test_data["Student_Query"]
y_test = test_data["Priority_Label"]

# Convert text into numbers

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Check accuracy

predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)

st.write("Model Accuracy:", round(accuracy * 100, 2), "%")

st.markdown("---")

# User Input

query = st.text_area("Enter Student Query")

if st.button("Predict"):
    if query != "":
        query_tfidf = vectorizer.transform([query])

    result = model.predict(query_tfidf)

    st.subheader("Predicted Priority")

    st.success(result[0])

else:
    st.warning("Please enter a query.")


st.markdown("---")

# Show Dataset

if st.checkbox("Show Dataset"):
    st.dataframe(train_data.head())


# Graph

st.subheader("Priority Distribution")

priority_count = train_data["Priority_Label"].value_counts()

st.bar_chart(priority_count)
