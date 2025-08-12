from flask import Flask, request, render_template
import pandas as pd
import joblib
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

# --- Tell NLTK where Render stores downloaded data ---
nltk.data.path.append("/opt/render/nltk_data")

app = Flask(__name__)

# --- Load model & vectorizer ---
model = joblib.load('model/model.pkl')
tfidf = joblib.load('model/tfidf.pkl')

# --- Initialize NLP tools ---
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()
ps = PorterStemmer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [ps.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

def preprocess_input_data(df):
    df = df.rename(columns={'requirements': 'job_requirements'})
    df['company_profile'] = df['company_profile'].fillna('').apply(preprocess_text)
    df['job_requirements'] = df['job_requirements'].fillna('').apply(preprocess_text)
    df['description'] = df['description'].fillna('').apply(preprocess_text)

    df['company_profile_word_count'] = df['company_profile'].apply(lambda x: len(word_tokenize(x)))
    df['job_requirements_word_count'] = df['job_requirements'].apply(lambda x: len(word_tokenize(x)))
    df['description_sentiment'] = df['description'].apply(lambda x: sia.polarity_scores(x)['compound'])
    df['job_requirements_sentiment'] = df['job_requirements'].apply(lambda x: sia.polarity_scores(x)['compound'])

    features = df[['company_profile_word_count', 'job_requirements_word_count',
                   'description_sentiment', 'job_requirements_sentiment']]
    tfidf_features = tfidf.transform(df['description']).toarray()
    tfidf_df = pd.DataFrame(tfidf_features, columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
    features = pd.concat([features.reset_index(drop=True), tfidf_df], axis=1)
    return features

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file and (file.filename.endswith('.csv') or file.filename.endswith('.txt')):
        df = pd.read_csv(file)
        required_columns = ['job_id', 'title', 'location', 'company_profile', 'description', 'requirements']
        if not all(col in df.columns for col in required_columns):
            return f"File must contain columns: {', '.join(required_columns)}", 400
        features = preprocess_input_data(df[['company_profile', 'description', 'requirements']])
        predictions = model.predict(features)
        df['Prediction'] = ['fraudulent' if pred == 1 else 'non-fraudulent' for pred in predictions]
        result_df = df[['job_id', 'title', 'location', 'Prediction']]
        return result_df.to_html(classes='table table-striped', index=False, escape=False)
    return "Invalid file format", 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
