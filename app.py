import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import ast


st.title("🔖 StackOverflow Tag Predictor")
st.markdown("Enter a programming question and description below to predict relevant StackOverflow tags.")


@st.cache_data
def model_data():
    df = pd.read_csv('clean.csv').sample(n=10000, random_state=42)
    df["tag_list"] = df["tag_list"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
   
    X = df['text']
    y = df['tag_list']
    mlb = MultiLabelBinarizer()
    y_mlb = mlb.fit_transform(y)
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_tfidf = tfidf.fit_transform(X)
    model = OneVsRestClassifier(LogisticRegression(solver='liblinear', class_weight='balanced'))
    model.fit(X_tfidf, y_mlb)

    return model, tfidf, mlb


model, tfidf, mlb = model_data()


question = st.text_area("💬 Enter your programming question:",height=100)
description=st.text_area('✏️ Enter Description :',height=100)
user_input = question + ' ' + description

def predict_tags(text):
    vector = tfidf.transform([text])
    proba = model.predict_proba(vector)[0]
    tags = []
    for i, score in enumerate(proba):
        if score >= 0.3:
            tags.append(mlb.classes_[i])
    return tags

if st.button("Predict Tags"):
    if user_input.strip():
        predict = predict_tags(user_input)
        if predict:
            pred_tags = ', '.join(predict)
            st.subheader("🏷️ Predicted Tags:")
            st.success(pred_tags)
        else:
            st.warning("No tags predicted")
    else:
        st.warning("Please enter text.")









