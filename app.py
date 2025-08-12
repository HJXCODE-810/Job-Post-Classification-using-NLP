from flask import Flask, request, render_template
import pandas as pd
import joblib
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import io

# ----------------------------------------------
# Download necessary NLTK resources (run once locally before deployment)
# Uncomment these lines if running for the first time
# nltk.download('punkt')
# nltk.download('vader_lexicon')
# nltk.download('stopwords')
# ----------------------------------------------

app = Flask(__name__)

# Load your pre-trained model and TF-IDF vectorizer
model = joblib.load('model/model.pkl')            # Adjust path as necessary
tfidf = joblib.load('model/tfidf.pkl')            # Adjust path as necessary

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()
ps = PorterStemmer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [ps.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(tokens)

def preprocess_input_data(df):
    """
    Takes in a DataFrame with columns: company_profile, description, requirements
    Returns a feature DataFrame ready for prediction
    """
    # Rename column for internal processing
    df = df.rename(columns={'requirements': 'job_requirements'})

    # Fill missing values and preprocess text
    df['company_profile'] = df['company_profile'].fillna('').apply(preprocess_text)
    df['job_requirements'] = df['job_requirements'].fillna('').apply(preprocess_text)
    df['description'] = df['description'].fillna('').apply(preprocess_text)

    # Calculate word counts and sentiment scores
    df['company_profile_word_count'] = df['company_profile'].apply(lambda x: len(word_tokenize(x)))
    df['job_requirements_word_count'] = df['job_requirements'].apply(lambda x: len(word_tokenize(x)))
    df['description_sentiment'] = df['description'].apply(lambda x: sia.polarity_scores(x)['compound'])
    df['job_requirements_sentiment'] = df['job_requirements'].apply(lambda x: sia.polarity_scores(x)['compound'])

    # Extract numeric features
    features = df[['company_profile_word_count', 'job_requirements_word_count',
                   'description_sentiment', 'job_requirements_sentiment']]
    
    # Transform description into TF-IDF features
    tfidf_features = tfidf.transform(df['description']).toarray()
    tfidf_df = pd.DataFrame(tfidf_features, columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
    
    # Combine numeric and TF-IDF features
    features = pd.concat([features.reset_index(drop=True), tfidf_df], axis=1)
    
    return features

@app.route('/')
def index():
    return render_template('index.html')  # Ensure index.html exists in 'templates' folder

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file", 400
    
    if file and (file.filename.endswith('.csv') or file.filename.endswith('.txt')):
        df = pd.read_csv(file)
        
        # Required columns in uploaded file
        required_columns = ['job_id', 'title', 'location', 'company_profile', 'description', 'requirements']
        if not all(col in df.columns for col in required_columns):
            return f"File must contain {', '.join(required_columns)} columns", 400
        
        # Preprocess data for model
        features = preprocess_input_data(df[['company_profile', 'description', 'requirements']])
        
        # Make predictions
        predictions = model.predict(features)
        
        # Add prediction result to DataFrame
        df['Prediction'] = ['fraudulent' if pred == 1 else 'non-fraudulent' for pred in predictions]
        
        # Prepare final result for display
        result_df = df[['job_id', 'title', 'location', 'Prediction']]
        return result_df.to_html(classes='table table-striped', index=False, escape=False)
        
    return "Invalid file format", 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
