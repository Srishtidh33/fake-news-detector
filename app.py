import streamlit as st
import pickle

# Load saved model
model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

st.title("📰 AI Fake News Detector")

st.write(
"This AI model analyses news articles and predicts whether they are REAL or FAKE using Machine Learning."
)

news_input = st.text_area("Enter News Article")

if st.button("Detect News"):

    if news_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        news_vector = vectorizer.transform([news_input])

        prediction = model.predict(news_vector)
        probabilities = model.predict_proba(news_vector)[0]

        # Map probabilities
        classes = model.classes_
        prob_dict = dict(zip(classes, probabilities))

        fake_prob = prob_dict.get("FAKE", 0)
        real_prob = prob_dict.get("REAL", 0)

        st.subheader("Result")

        if prediction[0] == "FAKE":
            st.error("⚠️ This news appears to be FAKE")
        else:
            st.success("✅ This news appears to be REAL")

        # Show confidence
        confidence = max(probabilities) * 100
        st.write("Confidence:", round(confidence, 2), "%")

        st.subheader("Prediction Probabilities")

        st.write("FAKE probability")
        st.progress(float(fake_prob))

        st.write("REAL probability")
        st.progress(float(real_prob))