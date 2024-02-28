# importing libraries
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import re
import numpy as np

def run():
    # buat header
    st.header("To our dearest customers:")
    
    # description
    st.write("Amazon would not be this big if it is not thanks to your endless support. Your valuable comments about our services and products would be greatly appreciated.")
    st.write("We are testing our latest model to classify your reviews in three sentiments: negative, neutral, and positive. If you are curious, go ahead and check it out!")

    # Load model yang sudah disimpan
    loaded_model = load_model('improved_model_tf')

#     # input all stuffs about customer
    with st.form("Input sample review text"):
        user_input = st.text_input("Please give your review about your recent transactions (at most 250 characters, please):", "",max_chars=250)
        # submit form
        sub = st.form_submit_button('Predict')

    if sub:
        data_predict = {'reviews.text': user_input}
        
        data = pd.DataFrame([data_predict])
        st.write("#### Input data result:")
        st.dataframe(data)
        try:
            # kolom isi stopwords
            stopwords = ['haven', 'only', 'why', 'o', 'his', 'won', 'how', "weren't", 'under', "mustn't", 'nor', 'which', 'himself', "you're", "aren't", 'shouldn', 'below', 'over', 'up', "shan't", 'than', "it's", 'your', 'against', 'down', "shouldn't", "you've", 'what', 'in', 'wasn', 'them', 'that', 'herself', 'not', 'does', 'had', 'for', 'during', 'further', 'him', 'itself', 'tv', 'isn', "couldn't", 'once', 'one', 'tablet', 'doesn', "wouldn't", 't', 'ourselves', 'before', 'app', 'until', 'while', 'couldn', 'an', 'fire', 'our', 'into', 'about', 'because', 'most', 'who', 'me', 'have', 'themselves', 'these', 'when', 'mustn', 'kid', 'ain', 'and', 'yours', 'do', 'i', "hasn't", 'yourselves', 'has', "she's", 'having', 'to', 've', 'again', 'their', 'some', 'own', 'whom', 'her', 'will', 'we', 'at', 'wouldn', 'other', 'needn', 'or', 'between', 'shan', 'don', 'being', 'd', 'same', 'by', 'with', 'm', 'as', 'from', 'hers', "should've", 's', "you'll", "mightn't", 'all', 'off', 'are', 'very', 'those', 'alexa', 'where', 'didn', 'mightn', 'few', 'but', 'can', 'am', "don't", 'hasn', 'hadn', 'weren', 'out', 'was', 'doing', "you'd", 'on', 'did', "didn't", 'it', "haven't", "needn't", 'no', "isn't", "won't", 'you', 'after', 'then', 'so', 'my', 'she', 'y', 'each', 'such', 'amazon', 'of', 'there', 'been', 'yourself', 'this', 'if', "hadn't", 'be', 'ma', 'just', 'the', 'its', 'more', 're', 'is', 'here', 'kindle', 'any', 'aren', 'echo', 'through', 'ours', 'theirs', 'were', "doesn't", 'now', "that'll", 'should', 'a', 'above', 'they', "wasn't", 'myself', 'both', 'too', 'll', 'he']

            # nltk
            nltk.download('punkt')

            # buat fungsi text preporcessing
            def text_preprocessing(filtered_dataset, stopwords):
                # Convert text in 'sentiment.value' column to lowercase
                text = filtered_dataset['reviews.text'].apply(lambda x: x.lower())

                # Mention removal
                text = text.apply(lambda x: re.sub("@[A-Za-z0-9_]+", " ", x))

                # Hashtags removal
                text = text.apply(lambda x: re.sub("#[A-Za-z0-9_]+", " ", x))

                # Newline removal (\n)
                text = text.apply(lambda x: re.sub(r"\\n", " ", x))

                # Whitespace removal
                text = text.apply(lambda x: x.strip())

                # URL removal
                text = text.apply(lambda x: re.sub(r"http\S+", " ", x))
                text = text.apply(lambda x: re.sub(r"www.\S+", " ", x))

                # Non-letter removal (such as emoticon, symbol (like μ, $, 兀), etc
                text = text.apply(lambda x: re.sub("[^A-Za-z\s']", " ", x))

                # Tokenization
                tokens = text.apply(lambda x: word_tokenize(x))

                # Stopwords removal
                tokens = tokens.apply(lambda x: [word for word in x if word not in stopwords])
                # Combining Tokens
                text = tokens.apply(lambda x: ' '.join(x))

                return text
            
            # implementasi fungsi
            data['reviews.text.processed'] = text_preprocessing(data, stopwords)

            # definisi X
            X_value = data['reviews.text.processed']

            # Perform inference using the loaded model
            prediction_results = loaded_model.predict(X_value)

            # Assuming the predictions are stored in a NumPy array 'predictions'
            prediction_array = np.array(prediction_results)

            # Find the index of the maximum value
            max_index = np.argmax(prediction_array)

            # Define sentiment labels
            sentiment_labels = ["negative", "neutral", "positive"]

            # Assign sentiment label based on the index of the maximum value
            sentiment = sentiment_labels[max_index]

            st.write('The review text given by the customer is classified with the sentiment type:', sentiment)
        except Exception as e:
            st.error(f'An error occurred: {e}')
    
            
if __name__ == '__main__':
    run()
