import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load and train model
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['message'])
y = df['label_num']

model = MultinomialNB()
model.fit(X, y)

# Streamlit UI
st.title("📩 Spam Message Classifier")
msg = st.text_area("Enter a message:")

if st.button("Predict"):
    msg_vector = vectorizer.transform([msg])
    pred = model.predict(msg_vector)[0]
    prob = model.predict_proba(msg_vector)[0][1]

    if pred == 1:
        st.error(f"❌ Spam message (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ Not spam (Confidence: {1 - prob:.2f})")
