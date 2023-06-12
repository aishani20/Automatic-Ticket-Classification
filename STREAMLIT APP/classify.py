# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 07:59:58 2023

@author: user
"""

import numpy as np
import pickle
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Loading the saved model
loaded_model = pickle.load(open('D:/Automatic_Ticket_Classification --NLP Project/logreg_model.pkl', 'rb'))

# Loading the CountVectorizer vocabulary
loaded_vec = CountVectorizer(vocabulary=pickle.load(open('D:/Automatic_Ticket_Classification --NLP Project/count_vector.pkl', 'rb')))
loaded_tfidf = pickle.load(open('D:/Automatic_Ticket_Classification --NLP Project/tfidf.pkl', 'rb'))

# Defining the target names
target_names = ["Bank Account services", "Credit card or prepaid card", "Others", "Theft/Dispute Reporting", "Mortgage/Loan"]



# Creating a function for topic prediction
def predict_topic(complaint):
    # Transform the input complaint into vector representation
    X_new_counts = loaded_vec.transform([complaint])
    X_new_tfidf = loaded_tfidf.transform(X_new_counts)

    # Make prediction using the loaded model
    predicted_topic = loaded_model.predict(X_new_tfidf)

    return target_names[predicted_topic[0]]

def main():
    # Giving a title
    st.title('Complaint Topic Prediction')
   

    # Getting the input complaint from the user
    complaint = st.text_area('Enter the complaint')

    # Code for prediction
    predicted_topic = ''
    if st.button('Predict Topic'):
        predicted_topic = predict_topic(complaint)

    # Displaying the predicted topic
    st.success('Predicted Topic: {}'.format(predicted_topic))

if __name__ == '__main__':
    main()
