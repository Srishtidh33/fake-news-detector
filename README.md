## Fake News Detection using Machine Learning

## Project Overview
This project uses Machine Learning and Natural Language Processing to detect fake news articles.
The model was trained using a labelled dataset of real and fake news articles. It uses TF-IDF vectorization for feature extraction and Logistic Regression for classification.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Streamlit

## Features
- Text preprocessing and cleaning
- Machine learning classification
- Model confidence score
- Interactive web application

## Model Performance
Accuracy: ~91%

## How to Run

1. Train the model
python fake_news_model.py

2. Run the web application
streamlit run app.py

## Live Demo

You can try the Fake News Detection app here:
https://fake-news-ml-detector.streamlit.app/

## Future Improvements
- Use transformer models like BERT
- Train on larger modern datasets
- Improve generalization across news domains
