
Job Post Classification Web Application

This repository contains a machine learning project aimed at classifying job posts into relevant categories using Natural Language Processing (NLP) and the Random Forest classification algorithm. The model is deployed as a web-based application using Flask, with model persistence facilitated through pickle.

---

## Table of Contents

- Overview
- Features
- Installation
- Usage
- Technologies Used
- Project Structure
- License

---

## Overview

The Job Post Classification Web Application is designed to categorize job postings based on the text content, helping users efficiently sort and search through various job descriptions. The project utilizes a Random Forest classifier with basic NLP preprocessing to achieve high accuracy in classification.

## Features

- Classifies job postings based on job description content
- Deployed as a web application for easy access and use
- Model is saved and loaded using pickle for efficient deployment and scaling

## Installation

1. **Clone the repository:**
   ```
   git clone https://github.com/HJXCODE-810/Job-Post-Classification-using-NLP.git
   cd job-post-classification
   ```

2. **Install dependencies:**
   Make sure you have Python installed. Then, install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```
   python app.py
   ```

4. Access the web application by navigating to `http://127.0.0.1:5000/` in your browser.

## Usage

1. **Upload Job Post Descriptions:** Input the job descriptions into the provided text box.
2. **View Classification Results:** The application will classify each job post into a predefined category and display the result.

## Technologies Used

- Python: Core programming language
- Flask: Web framework for deployment
- Random Forest Classifier: Machine Learning model for classification
- NLP (Natural Language Processing): For preprocessing job description text
- Pickle: For model persistence

## Project Structure

- `app.py`: Flask application script
- `model/`: Directory containing the trained model files
- `templates/`: HTML templates for the web interface
- `static/`: CSS and JavaScript files
- `requirements.txt`: List of required Python packages



---

Feel free to contribute, raise issues, or request features. Enjoy exploring job post classification with this application!
