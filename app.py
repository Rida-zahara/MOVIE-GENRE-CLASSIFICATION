import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# --- PAGE CONFIG ---
st.set_page_config(page_title="Movie Genre Predictor", page_icon="🎬", layout="centered")

# --- CACHED MODEL LOADING ---
@st.cache_resource
def train_model():
    # Loading data from your path
    path = 'data/train_data.txt' 
# This tells Render to look inside the 'data' folder in YOUR REPO"
    cols = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION']
    df = pd.read_csv(path, sep=':::', names=cols, engine='python').head(20000) # Using 20k for speed
    
    # Simple Clean
    stop_words = set(stopwords.words('english'))
    def clean(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = word_tokenize(text)
        return " ".join([w for w in tokens if w not in stop_words])

    df['CLEAN'] = df['DESCRIPTION'].apply(clean)
    
    # Vectorize
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X = tfidf.fit_transform(df['CLEAN'])
    y = df['GENRE'].str.strip()
    
    # Train
    model = LinearSVC()
    model.fit(X, y)
    return tfidf, model, clean

# UI Logic
st.title("🎬 Movie Genre Classifier")
st.markdown("Enter a movie plot summary below to predict its genre.")

with st.spinner("Loading AI Model..."):
    tfidf, model, clean_func = train_model()

# User Input
plot_input = st.text_area("Movie Plot Summary:", placeholder="e.g., A brave warrior leads his army against an alien invasion...", height=150)

if st.button("Predict Genre"):
    if plot_input:
        cleaned = clean_func(plot_input)
        vec = tfidf.transform([cleaned])
        prediction = model.predict(vec)[0]
        
        # Display Result
        st.success(f"### Predicted Genre: **{prediction}**")
        st.balloons()
    else:
        st.warning("Please enter a plot first!")