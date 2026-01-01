import pandas as pd
import re
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# MAC SSL CERTIFICATE FIX 
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# SETUP & NLTK RESOURCES
print("--- Initializing NLTK Resources ---")
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# CONFIGURATION (Your specific paths) 
TRAIN_PATH = "/Users/rida/Documents/movie_genre_classification/data/train_data.txt"
TEST_SOL_PATH = "/Users/rida/Documents/movie_genre_classification/data/test_data_solution.txt"

# PREPROCESSING FUNCTION 
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    # Remove stopwords and non-alphabetic characters
    cleaned = [w for w in tokens if w not in stop_words and w.isalpha()]
    return " ".join(cleaned)

# LOAD DATA
print("--- Loading Datasets ---")
cols = ['ID', 'TITLE', 'GENRE', 'DESCRIPTION']
try:
    train_df = pd.read_csv(TRAIN_PATH, sep=':::', names=cols, engine='python')
    test_df = pd.read_csv(TEST_SOL_PATH, sep=':::', names=cols, engine='python')
    print(f"Successfully loaded {len(train_df)} training samples.")
except FileNotFoundError as e:
    print(f"Error: Could not find files. Check your paths.\n{e}")
    exit()

# CLEANING
print("--- Cleaning Descriptions (This may take a minute) ---")
train_df['CLEAN_DESCRIPTION'] = train_df['DESCRIPTION'].apply(clean_text)
test_df['CLEAN_DESCRIPTION'] = test_df['DESCRIPTION'].apply(clean_text)

# VECTORIZATION (TF-IDF)
print("--- Vectorizing Text ---")
# Using 10,000 features and bigrams (pairs of words) for better context
tfidf = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1,2))
X_train = tfidf.fit_transform(train_df['CLEAN_DESCRIPTION'])
X_test = tfidf.transform(test_df['CLEAN_DESCRIPTION'])

y_train = train_df['GENRE'].str.strip()
y_test = test_df['GENRE'].str.strip()

# MODEL TRAINING (Linear SVM) 
print("--- Training Linear SVC Model ---")
model = LinearSVC(C=1.0, max_iter=1000)
model.fit(X_train, y_train)

# EVALUATION 
print("\n" + "="*40)
print("         MODEL EVALUATION")
print("="*40)
y_pred = model.predict(X_test)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report (Key Genres):")
print(classification_report(y_test, y_pred, zero_division=0))

# PREDICTION TEST 
def predict_movie(plot):
    cleaned = clean_text(plot)
    vec = tfidf.transform([cleaned])
    return model.predict(vec)[0]

print("\n--- Manual Prediction Test ---")
sample_plot = "A futuristic scientist discovers a way to travel back in time to prevent a global disaster."
result = predict_movie(sample_plot)
print(f"Plot: {sample_plot}")
print(f"Predicted Genre: {result}")