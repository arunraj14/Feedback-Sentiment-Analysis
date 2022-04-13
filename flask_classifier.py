# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 2022

@author: Arunraj
"""

from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import nltk as nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

# unpickling the model
pickle_in = open("multi_classifier.pkl","rb")
classifier=pickle.load(pickle_in)

# label for prediction Output
pred = {0: 'Positive', 1: 'Neutral',2:'Negative'}

# data preprocessing
def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """    
    lemmatizer = WordNetLemmatizer()
    all_stopwords = stopwords.words('english')
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    stop_words = set(stopwords.words('english'))

    text = BeautifulSoup(text, "html.parser").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in stop_words) # delete stopwords from text
    text = ' '.join([lemmatizer.lemmatize(word,'v') for word in text.split()]) # Use lemmatizer to identify root word
    return text

@app.route('/')
def welcome():
    return "Please use localhost:8000/apidocs Or 127.0.0.1:8000/apidocs/ for Prediction"

@app.route('/predict',methods=["Get"])
def predict_note_authentication():
    
    """Let's predict Health Authority feedback sentiments 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: feedback
        in: query
        type: string
        required: true
    responses:
        200:
            description: The output values
        
    """
    feedback=request.args.get("feedback")
    rev_pre = clean_text(feedback)
    prediction=classifier.predict([rev_pre])    
    return "Hello The answer is "+str(pred[prediction[0]])

@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    """Let's predict Health Authority feedback sentiments 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    rev_df=pd.read_csv(request.files.get("file"),header=None)
    rev_df.rename({0:'Sentence'},axis=1,inplace=True)
    if rev_df.iloc[0]['Sentence'] == ('Sentence' or 'sentence'):
        rev_df = rev_df.drop(0)
    rev_df['Sentence'] = rev_df['Sentence'].apply(clean_text)
    prediction=classifier.predict(rev_df['Sentence'])
        
    return str(list(prediction))

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)
       