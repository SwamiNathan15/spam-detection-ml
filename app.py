# app.py
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os

st.set_page_config(page_title="Spam Detector", layout="centered")

st.title("📩 Email / SMS Spam Detector")
st.write("Type a message and the model will predict whether it's spam or not.")

DATA_CSV = "spam.csv"
RAW_FILE = "SMSSpamCollection"
MODEL_FILE = "spam_model.joblib"
VECT_FILE = "tfidf_vectorizer.joblib"

@st.cache_data(show_spinner=False)
def load_dataset():
    # Try csv first
    if os.path.exists(DATA_CSV):
        df = pd.read_csv(DATA_CSV)
    elif os.path.exists(RAW_FILE):
        df = pd.read_csv(RAW_FILE, sep="\t", header=None, names=["label","text"])
        df.to_csv(DATA_CSV, index=False)
    else:
        return None
    # ensure label numeric column exists
    if "label_num" not in df.columns:
        df["label_num"] = df.label.map({"ham": 0, "spam": 1})
    # ensure text column string
    df["text"] = df["text"].astype(str)
    return df

@st.cache_resource(show_spinner=False)
def train_model(df):
    # tfidf + MultinomialNB trained on full dataset
    tfidf = TfidfVectorizer(stop_words="english")
    X = tfidf.fit_transform(df["text"])
    y = df["label_num"].values
    model = MultinomialNB()
    model.fit(X, y)
    # optionally save
    joblib.dump(model, MODEL_FILE)
    joblib.dump(tfidf, VECT_FILE)
    return model, tfidf

def load_or_train():
    # if saved model exists, load it; else train
    if os.path.exists(MODEL_FILE) and os.path.exists(VECT_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            tfidf = joblib.load(VECT_FILE)
            return model, tfidf, "loaded"
        except Exception:
            # if loading fails, fall back to training
            pass
    df = load_dataset()
    if df is None:
        return None, None, "no_data"
    model, tfidf = train_model(df)
    return model, tfidf, "trained"

model, tfidf, status = load_or_train()

if status == "no_data":
    st.error("Could not find a dataset. Please place 'spam.csv' or 'SMSSpamCollection' in this folder.")
    st.markdown("""
    **How to create `spam.csv`:**
    - You already have `SMSSpamCollection` from the UCI SMS Spam Collection zip.
    - Upload it to this folder and the app will convert it automatically.
    """)
    st.stop()

if status == "trained":
    st.success("Model trained on local dataset (saved to disk).")
elif status == "loaded":
    st.info("Loaded saved model and vectorizer from disk.")

# Left column: input
st.subheader("Try it now")
user_text = st.text_area("Type your message here", height=120, placeholder="e.g. Free entry! Click here to claim your prize...")
if st.button("Predict"):
    if not user_text or user_text.strip()=="":
        st.warning("Please type a message first.")
    else:
        vect = tfidf.transform([user_text])
        pred = model.predict(vect)[0]
        probs = model.predict_proba(vect)[0]
        spam_prob = probs[1]
        if int(pred) == 1:
            st.error(f"SPAM 🚫  (probability {spam_prob:.2f})")
        else:
            st.success(f"NOT SPAM ✅  (spam probability {spam_prob:.2f})")

# Right / below: examples & model info
st.markdown("---")
st.subheader("Quick examples")
col1, col2 = st.columns(2)
with col1:
    if st.button("Example: Free win"):
        st.experimental_set_query_params()
        example = "Congratulations! You won a free iPhone. Click here to claim."
        st.session_state["__example__"] = example
with col2:
    if st.button("Example: Normal"):
        example = "Hey, are you coming to class tomorrow?"
        st.session_state["__example__"] = example

# If an example was chosen, fill the text area (works around to update)
if "__example__" in st.session_state:
    st.write("Example message inserted below — press Predict.")
    st.text_area("Example", value=st.session_state["__example__"], height=100, key="example_box")
    # copy to main text area would require rerun; user can copy/paste or use the example box

st.markdown("---")
st.write("Model and vectorizer status:")
st.code(f"Model file: {os.path.exists(MODEL_FILE)}  Vectorizer file: {os.path.exists(VECT_FILE)}")
st.caption("Model: MultinomialNB trained on TF-IDF of the SMS Spam dataset.")

st.markdown("**Note:** If you want to retrain the model on a different dataset, replace `spam.csv` and restart the app.")
