import streamlit as st
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout_rate=0.3):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        output = self.dropout(output[:, -1, :])
        output = self.fc(output)
        return output

# Load model and tokenizer
device = torch.device('cpu')

with open("word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)
with open("idx2word.pkl", "rb") as f:
    idx2word = pickle.load(f)

vocab_size = len(word2idx)
embed_size = 128
hidden_size = 256

model = LSTMModel(vocab_size, embed_size, hidden_size)
model.load_state_dict(torch.load("lstm_predictive_model.pth", map_location=device))
model.eval()


# Streamlit app
# Setup Streamlit page
st.set_page_config(
    page_title="Smart Email Composer",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .main {
            background-color: #f9f9f9;
        }
        .title {
            text-align: center;
            color: #4B4B4B;
            font-size: 36px;
            font-weight: 700;
            margin-bottom: 20px;
        }
        .subtitle {
            text-align: center;
            color: #6e6e6e;
            font-size: 18px;
            margin-bottom: 40px;
        }
        .result {
            background-color: #e8f0fe;
            padding: 15px;
            border-radius: 8px;
            font-size: 20px;
            color: #202124;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<div class="title">📧 Smart Email Composer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict the next word in your sentence using AI-powered LSTM intelligence.</div>',
    unsafe_allow_html=True,
)

# Input
input_text = st.text_input(
    "Type your email content here...",
    placeholder="e.g., Thank you for your email. I would like to...",
    help="Enter a partial sentence and let the AI predict the next word."
)

# Prediction logic
def predict_next_word(text):
    tokens = [word2idx.get(w.lower(), word2idx.get("<UNK>", 0)) for w in text.strip().split()]
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        next_token = torch.argmax(output, dim=1).item()
        next_word = idx2word.get(next_token, "<UNK>")

    return next_word

# Action
if st.button("✏️ Predict Next Word"):
    if input_text.strip():
        next_word = predict_next_word(input_text)
        st.markdown(f"<div class='result'><strong>Next word:</strong> {next_word}</div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to continue.")

# About section
st.markdown("---")
st.markdown("### About this App")
st.write(
    """
    The Smart Email Composer uses an LSTM neural network to help users predict the next word while drafting emails.
    It’s a useful tool for productivity, especially when composing formal or business communication.
    """
)

st.caption("Crafted with ❤️ by Yagna")
