import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the model and tokenizer
model = load_model('spam_lstm_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

st.title("Spam Email Classifier")

message = st.text_area("Enter your email message here:")

if st.button("Classify"):
    if message:
        message_seq = tokenizer.texts_to_sequences([message])
        message_padded = pad_sequences(message_seq, maxlen=100, padding='post', truncating='post')
        prediction = model.predict(message_padded)
        spam_probability = prediction[0][0]
        result = "Spam" if spam_probability > 0.5 else "Not Spam"
        st.write(f"Prediction: {result}")
        st.write(f"Spam Probability: {spam_probability * 100:.2f}%")
    else:
        st.write("Please enter a message to classify.")
