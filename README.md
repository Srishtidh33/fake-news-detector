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

## Demo Screenshot
<img width="873" height="771" alt="image" src="https://github.com/user-attachments/assets/23fb9a88-c221-4766-8aa5-7caff93b9302" />

## Run Locally
Clone the project:

git clone https://github.com/yourusername/fake-news-detection-ml.git

Install dependencies:

pip install -r requirements.txt

Run the app:

streamlit run app.py

## Limitations
- Model trained on older political dataset (2016 US election)
- May misclassify modern or non-political news
- Limited generalization to unseen domains

## Future Improvements
- Use transformer models like BERT
- Train on larger modern datasets
- Improve generalization across news domains
