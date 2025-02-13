import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load translation models and tokenizers
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Change the models to translate FROM French/Russian TO English
translator_models = {
    "French → English": "Helsinki-NLP/opus-mt-fr-en",
    "Russian → English": "Helsinki-NLP/opus-mt-ru-en"
}

tokenizers_models = {lang: load_model(model) for lang, model in translator_models.items()}

def translate_text(text, lang):
    tokenizer, model = tokenizers_models[lang]
    inputs = tokenizer(text, return_tensors="pt")
    translated_ids = model.generate(**inputs)
    return tokenizer.decode(translated_ids[0], skip_special_tokens=True)

# Streamlit UI
st.title("Multilingual Translation Bot")
st.write("Translate French or Russian text to English.")

text_to_translate = st.text_area("Enter text in French or Russian:", "")
language = st.selectbox("Select source language:", ["French → English", "Russian → English"])

if st.button("Translate"):
    if text_to_translate.strip():
        translated_text = translate_text(text_to_translate, language)
        st.success(f"Translation ({language}): {translated_text}")
    else:
        st.warning("Please enter some text to translate.")
