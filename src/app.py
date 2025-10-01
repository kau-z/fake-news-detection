import streamlit as st
import json
import joblib
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import shap

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Bidirectional

# ========================
# Setup paths
# ========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models", "final")
TOKENIZER_PATH = os.path.join(BASE_DIR, "..", "models", "tokenizer.pkl")

# ========================
# Load Metadata + Model
# ========================
with open(os.path.join(MODELS_DIR, "metadata.json")) as f:
    meta = json.load(f)

try:
    model = load_model(os.path.join(MODELS_DIR, "best_lstm_bilstm.keras"), compile=False)
except Exception as e:
    print("⚠️ .keras model failed, trying .h5 instead:", e)
    model = load_model(
        os.path.join(MODELS_DIR, "best_lstm_bilstm.h5"),
        custom_objects={"LSTM": LSTM, "Bidirectional": Bidirectional},
        compile=False
    )

tokenizer = joblib.load(TOKENIZER_PATH)
MAX_LEN = meta["preprocessing"]["max_len"]

# ========================
# Prediction Functions
# ========================
def predict_text(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    prob = model.predict(pad, verbose=0)[0][0]
    label = "📰 Real News" if prob < 0.5 else "⚠️ Fake News"
    return label, float(prob), pad

def predict_batch(texts):
    seqs = tokenizer.texts_to_sequences(texts)
    pads = pad_sequences(seqs, maxlen=MAX_LEN, padding="post")
    probs = model.predict(pads, verbose=0).ravel()
    labels = ["📰 Real News" if p < 0.5 else "⚠️ Fake News" for p in probs]
    return pd.DataFrame({
        "Text": texts,
        "Prediction": labels,
        "Fake Probability": probs
    })

# ========================
# Streamlit UI
# ========================
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection App")
st.write("Detect if news is **Real or Fake** using a deep learning model. Now with Explainable AI 🧠")

tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📂 Batch Prediction (CSV)", "🧠 Explain Prediction"])

# --- Single Prediction ---
with tab1:
    user_input = st.text_area("✍️ Enter news text:", height=150)

    if st.button("Analyze", key="single"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            label, prob, _ = predict_text(user_input)
            st.subheader("Prediction:")
            st.write(f"**{label}**")
            st.write(f"Confidence (Fake Probability): `{prob:.2f}`")

# --- Batch Prediction ---
with tab2:
    uploaded_file = st.file_uploader("📂 Upload a CSV file with 'text' or 'description' column", type=["csv"])
    
    if uploaded_file is not None:
        df = None
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(uploaded_file, encoding="latin-1")
            except Exception as e:
                st.error(f"Could not read CSV file: {e}")

        if df is not None:
            text_col = None
            if "text" in df.columns:
                text_col = "text"
            elif "description" in df.columns:
                text_col = "description"

            if text_col is None:
                st.error("CSV must have a column named 'text' or 'description'")
            else:
                st.write("✅ File uploaded successfully! Preview:")
                st.dataframe(df.head())

                if st.button("Run Batch Prediction", key="batch"):
                    results = predict_batch(df[text_col].astype(str).tolist())
                    st.subheader("Batch Prediction Results")

                    styled = results.style.background_gradient(
                        cmap="RdYlGn_r", subset=["Fake Probability"]
                    )
                    st.dataframe(styled, use_container_width=True)

                    st.subheader("📊 Prediction Distribution")
                    counts = results["Prediction"].value_counts()
                    
                    fig, ax = plt.subplots(figsize=(5,3))
                    counts.plot(kind="bar", ax=ax, color=["green", "red"])
                    ax.set_ylabel("Count")
                    ax.set_title("Fake vs Real News Predictions")
                    st.pyplot(fig)

                    csv = results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name="batch_predictions.csv",
                        mime="text/csv"
                    )

# --- Explain Prediction ---
with tab3:
    explain_text = st.text_area("✍️ Enter news text to explain:", height=150)

    if st.button("Explain", key="explain"):
        if explain_text.strip() == "":
            st.warning("Please enter some text.")
        else:
            label, prob, _ = predict_text(explain_text)

            st.subheader("Prediction:")
            st.write(f"**{label}** (Confidence: `{prob:.2f}`)")

            # ==============================
            # LIME Text Explainer
            # ==============================
            from lime.lime_text import LimeTextExplainer

            explainer = LimeTextExplainer(class_names=["Real", "Fake"])

            def predict_lime(texts):
                seqs = tokenizer.texts_to_sequences(texts)
                pads = pad_sequences(seqs, maxlen=MAX_LEN, padding="post")
                preds = model.predict(pads, verbose=0)
                return np.hstack([1 - preds, preds])  # shape: [N,2]

            exp = explainer.explain_instance(
                explain_text,
                predict_lime,
                num_features=10
            )

            # Show table of contributions
            st.subheader("🔍 Top Word Contributions")
            contribs = pd.DataFrame(exp.as_list(), columns=["Word", "Impact"])
            st.dataframe(contribs)

            # Bar chart
            st.subheader("📊 Contribution Strength")
            fig, ax = plt.subplots(figsize=(6, 3))
            contribs.set_index("Word").Impact.plot(
                kind="bar", ax=ax, color="purple"
            )
            ax.set_title("Word Impact on Prediction")
            ax.set_ylabel("Contribution")
            st.pyplot(fig)

            # Highlighted explanation (custom Streamlit safe)
            st.subheader("🖍️ Highlighted Explanation")
            highlighted_text = ""
            for word, impact in exp.as_list():
                intensity = min(1, abs(impact) / max(abs(i) for _, i in exp.as_list()))  # normalize 0–1
                if impact > 0:
                    color = f"rgba(255, 0, 0, {intensity})"  # red → fake
                else:
                    color = f"rgba(0, 255, 0, {intensity})"  # green → real
                highlighted_text += f"<span style='background-color:{color}; padding:2px; margin:1px'>{word}</span> "

            st.markdown(highlighted_text, unsafe_allow_html=True)

