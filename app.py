#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:26:34 2021

@author: sharad
"""


from flask import Flask,request
import pandas as pd
import numpy as np
import pickle 
import streamlit as st
from PIL import Image


pickle_in=open('language_predictor.pkl','rb')
language_predictor=pickle.load(pickle_in)

cv_pickle=open('vectorize.pkl','rb')
cv=pickle.load(cv_pickle)

le_pickle=open('encoder.pkl','rb')
le=pickle.load(le_pickle)
              
def lang_predict(text):
     x = cv.transform([text]).toarray() # converting text to bag of words model (Vector)
     lang = language_predictor.predict(x) # predicting the language
     lang = le.inverse_transform(lang) # finding the language corresponding the the predicted value
     return lang[0] # return the predicted language

def main():
    st.title("Language Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Language Predictor using NLP</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    text=st.text_input("Text to Predict","Type Here")
    result=""
    if st.button("Predict"):
        result=lang_predict(text)
    st.success('The given text is written in {}'.format(result))
    if st.button("About"):
        st.text("Predicting Language of a given text using NLP")
        st.text("API built with Streamlit")
        

if __name__== '__main__':
    main()