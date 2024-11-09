from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load the model, encoder, and TF-IDF Vectorizer
with open('E:/Assignments/0.1_web_dev/model/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('E:/Assignments/0.1_web_dev/model/encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)
with open('E:/Assignments/0.1_web_dev/model/tfidf.pkl', 'rb') as tfidf_file:
    tfidf = pickle.load(tfidf_file)

# Text preprocessing function
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [ps.stem(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure TF-IDF, encoder, and model are loaded
        global tfidf, encoder, model
        if tfidf is None or encoder is None or model is None:
            return jsonify({'error': 'Model, TF-IDF, or encoder not initialized.'}), 500

        # Get data from request form
        data = request.form
        required_fields = ['company_profile', 'description', 'requirements', 'benefits',
                           'telecommuting', 'has_company_logo', 'has_questions',
                           'required_experience', 'required_education', 'industry', 'function']

        # Check for missing fields
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # Strip leading and trailing spaces from specified fields
        clean_data = {field: data[field].strip() if field in ['required_experience', 'required_education', 'industry', 'function'] else data[field]
                      for field in required_fields}

        # Preprocess and combine text fields
        input_text = ' '.join([preprocess_text(clean_data[col]) for col in ['company_profile', 'description', 'requirements', 'benefits']])

        # Transform input text with TF-IDF
        X_text = tfidf.transform([input_text])

        # Prepare additional features
        additional_features = pd.DataFrame({
            'telecommuting': [int(clean_data['telecommuting'])],
            'has_company_logo': [int(clean_data['has_company_logo'])],
            'has_questions': [int(clean_data['has_questions'])],
            'required_experience': [clean_data['required_experience']],
            'required_education': [clean_data['required_education']],
            'industry': [clean_data['industry']],
            'function': [clean_data['function']]
        })

        # Encode categorical features
        encoded_features = encoder.transform(additional_features[['required_experience', 'required_education', 'industry', 'function']])
        encoded_features_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

        # Combine all features into one DataFrame
        final_input = pd.concat(
            [pd.DataFrame(X_text.toarray()), additional_features[['telecommuting', 'has_company_logo', 'has_questions']], encoded_features_df],
            axis=1
        )

        # Ensure columns match the model’s expected input
        final_input = final_input.reindex(columns=model.feature_names_in_, fill_value=0)

        # Make prediction
        prediction = model.predict(final_input)[0]
        result = "Fraudulent" if prediction == 1 else "Legitimate"

        return jsonify({'prediction': result})

    except Exception as e:
        # Log the error if needed
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)
