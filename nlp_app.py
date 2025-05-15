import streamlit as st
import pandas as pd
from textblob import TextBlob
from pickle import load

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
stemmer=PorterStemmer()
stop_words=stopwords.words('english')
def preprs(text):
    text=text.lower()
    text=re.sub(r'[^a-zA-Z\s]','',text)
    text=re.sub(f'[{string.punctuation}]','',text)
    tokens=word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

model_bundle = load(open('nlp.pkl', 'rb'))
model = model_bundle['model']
vectorizer = model_bundle['vectorizer']

st.title('Sentiment analysis on Amazon reviews')
input_txt=st.text_area('Enter a Review')

if st.button('Predict'):
	if not input_txt.strip():
		st.warning('Please enter a review')
	else:	
		processed=preprs(input_txt)
		vec_in=vectorizer.transform([processed])
		predict=model.predict(vec_in)[0]
		sentiment=TextBlob(input_txt).sentiment.polarity

		if sentiment > 0:
			st.write('Reviwe is **positive**')
		elif sentiment < 0:
			st.write('Review is **Negative**')
		else:
			st.write('It is  **Neutral**')