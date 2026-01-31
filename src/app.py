import streamlit as st
import pickle
import re
import spacy
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="Clasificador y Recomendador Musical",
    page_icon="🎵"
)


@st.cache_resource
def load_models():
    with open("src/model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("src/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("src/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return model, vectorizer, label_encoder

model, vectorizer, label_encoder = load_models()


@st.cache_resource
def load_nlp():
    try:
        return spacy.load("es_core_news_sm")
    except OSError:
        from spacy.cli import download
        download("es_core_news_sm")
        return spacy.load("es_core_news_sm")

nlp = load_nlp()

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
    text = re.sub(r'\s+', ' ', text).strip()
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


@st.cache_data
def load_dataset_and_vectors():
    df = pd.read_csv("src/lyrics,label.txt")
    df = df.dropna(subset=["lyrics", "label"])

    df["lyrics_clean"] = df["lyrics"].apply(
        lambda x: clean_text(pre_clean(x))
    )

    vectors = vectorizer.transform(df["lyrics_clean"])
    return df, vectors

df, lyrics_vectors = load_dataset_and_vectors()


def predict_genre(text):
    text_clean = clean_text(pre_clean(text))
    vec = vectorizer.transform([text_clean])
    pred = model.predict(vec)
    return label_encoder.inverse_transform(pred)[0]


def recommend_song(text, genre):
    text_clean = clean_text(pre_clean(text))
    user_vec = vectorizer.transform([text_clean])

    df_genre = df[df["label"] == genre]
    vectors_genre = vectorizer.transform(df_genre["lyrics_clean"])

    similarities = cosine_similarity(user_vec, vectors_genre)[0]
    best_idx = similarities.argmax()

    return df_genre.iloc[best_idx]["lyrics"]


st.title("🎶 Clasificador y Recomendador Musical")

st.write(
    "Escribe una frase o parte de una letra y el sistema:\n"
    "- 🎼 Predice el **género musical**\n"
    "- 🎵 Recomienda una **canción similar por contexto y sentimiento**"
)

user_input = st.text_area(
    "Letra de la canción",
    height=200,
    placeholder="Ej: corazón no me abandones, que sin tu amor no sé vivir..."
)

if st.button("🎧 Analizar"):
    if user_input.strip() == "":
        st.warning("Por favor ingresa una letra.")
    else:
        with st.spinner("Analizando letra y buscando canciones similares... 🎶"):
            genre = predict_genre(user_input)
            recommended_lyrics = recommend_song(user_input, genre)

        st.success(f"🎼 Género predicho: **{genre}**")

        st.markdown("---")
        st.subheader("🎵 ¿Te gustaría escuchar algo como esto?")

        st.info(
            recommended_lyrics[:500] + "..."
            if len(recommended_lyrics) > 500
            else recommended_lyrics
        )
