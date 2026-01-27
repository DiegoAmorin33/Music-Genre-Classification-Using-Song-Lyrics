import streamlit as st
import pickle
import re
import spacy

with open("src/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("src/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("src/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

nlp = spacy.load("es_core_news_sm")

expresion_stopwords = {
    "oh","eh","uh","ay","yeh","yeah","oh oh",
    "yo","ey","ah","ah ah"
}

def pre_clean(text):
    text = text.lower()
    text = re.sub(r"[',]", "", text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub('bis', ' ', text)
    text = re.sub('[()]', ' ', text)
    text = re.sub('"', ' ', text)
    text = re.sub('\s+', ' ', text).strip()
    return text

def clean_text(text):
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and token.lemma_ not in expresion_stopwords
        and len(token.lemma_) > 2
    ]
    return " ".join(tokens)

def predict_genre(text):
    text_clean = clean_text(pre_clean(text))
    vec = vectorizer.transform([text_clean])
    pred = model.predict(vec)
    return label_encoder.inverse_transform(pred)[0]

st.set_page_config(page_title="Clasificador de Géneros Musicales", page_icon="🎵")

st.title("🎶 Clasificador de Géneros Musicales")
st.write(
    "Ingresa una frase o parte de la letra de una canción "
    "y el modelo predecirá el **género musical**."
)

user_input = st.text_area(
    "Letra de la canción",
    height=200,
    placeholder="Ej: corazón no me abandones, que sin tu amor no sé vivir..."
)

if st.button("🎧 Predecir género"):
    if user_input.strip() == "":
        st.warning("Por favor ingresa una letra.")
    else:
        genre = predict_genre(user_input)
        st.success(f"🎼 Género predicho: **{genre}**")
