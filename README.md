📰 Fake News Detection

A deep learning–powered Fake News Detection System that classifies news as Real or Fake, with support for Explainable AI. Built using LSTM + BiLSTM models, GloVe embeddings, and a Streamlit app for easy interaction.

🚀 Features

✅ Multiple ML/DL models (Logistic Regression, Random Forest, Naive Bayes, LSTM, BiLSTM)

✅ Word embeddings using GloVe

✅ Explainable AI with SHAP for word-level insights

✅ Streamlit App with:

Single prediction

Batch prediction (CSV upload)

Explanation of model decision

✅ Exportable results (CSV & charts)

📂 Project Structure
fake-news-detection/
│── data/                  # Raw & processed data (ignored in git)
│── glove/                 # GloVe embeddings (ignored in git)
│── models/                # Saved models & tokenizer (ignored in git)
│── notebooks/             # Jupyter notebooks (EDA, training, evaluation)
│── reports/               # Confusion matrices, comparisons (ignored in git)
│── src/                   
│   └── app.py             # Streamlit app
│── requirements.txt       # Dependencies
│── .gitignore             # Git ignore file
│── README.md              # Project documentation

⚙️ Installation

Clone the repo

git clone https://github.com/USERNAME/fake-news-detection.git
cd fake-news-detection


Create & activate virtual environment

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows


Install dependencies

pip install -r requirements.txt


Download GloVe embeddings
Place required glove.6B.*.txt files inside glove/.
👉 Download from Stanford NLP

▶️ Running the App

Run the Streamlit app:

streamlit run src/app.py

📊 Example Output

Prediction Distribution


Word Contributions (Explainable AI)
Highlighting most influential words in predictions.

🧠 Models Used

Logistic Regression

Random Forest

Naive Bayes

LSTM

BiLSTM

🔮 Future Improvements

Add transformer models (BERT, DistilBERT)

Support multi-language detection

Deploy on Heroku/Streamlit Cloud