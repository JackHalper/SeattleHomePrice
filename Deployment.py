#!/usr/bin/env python
# coding: utf-8

# # Redfin Listing Price Predictor

# ## Instructions

# Go To Redfin.com and find the listed property you want to predict 
# - 1. Navigate to Cell in the Toolbar
# - 2. Select Run All
# - 3. Input House Information 
# - 4. Copy and Paste House Description 
# - 5. Input House Listing Price 
# - 6. Scroll to the Bottom to Get Predictions
# 
# - PROPERTY TYPE INPUT OPTIONS (CASE SENSITIVE) 
#     - Single Family Residential
#     - Condo/Co-op	
#     - Townhouse

# In[686]:


get_ipython().system('pip install streamlit')


# In[658]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import spacy 
import re
from gensim.models.phrases import Phraser, Phrases
import pickle


# In[659]:


current_year = datetime.now().year
current_month = datetime.now().month
current_quarter = (current_month - 1) // 3 + 1  # Calculate the quarter based on the current month
current_date = datetime.now().date()

columns = [
    "Year",
    "Quarter",
    "YEAR BUILT",
    "SALE TYPE",
    "SOLD DATE",
    "PROPERTY TYPE",
    "ZIP OR POSTAL CODE",
    "BEDS",
    "BATHS",
    "LOCATION",
    "SQUARE FEET",
    "LOT SIZE", 
    "Description"
]

df = pd.DataFrame(columns=columns)
for column in columns:
    if column == "Year":
        user_input = current_year
    elif column == "Quarter":
        user_input = current_quarter
    elif column == "SOLD DATE":
        user_input = current_date 
    elif column == "SALE TYPE":
        user_input = "PAST SALE"  
    else:
        user_input = input(f"Enter {column}: ")
    df.at[0, column] = user_input
price = input(f"Enter Price")


# In[660]:


df["Age"] = 2023 - df["YEAR BUILT"].astype("int")
df["Year"] = df["Year"].astype("int")
df["ZIP OR POSTAL CODE"] = df["ZIP OR POSTAL CODE"].astype("int")
df["BEDS"] = df["BEDS"].astype("float")
df["BATHS"] = df["BATHS"].astype("float")
df["SQUARE FEET"] = df["SQUARE FEET"].astype("int")
df["LOT SIZE"] = df["LOT SIZE"].astype("int")
df["Quarter"] = df["Quarter"].astype("int")
df["SOLD DATE"] = pd.to_datetime(df["SOLD DATE"])
df["ZIP OR POSTAL CODE"] = df["ZIP OR POSTAL CODE"].astype("int")


# In[661]:


df.drop("YEAR BUILT", axis =1)


# In[662]:


df ["Description"] = df["Description"].str.lower()


# In[663]:


nlp = spacy.load('en_core_web_sm')
def lemmatize(text): 
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])
    return lemmatized_text


# In[664]:


df["Description"] = df["Description"].apply(lemmatize)


# In[665]:


stop_words = set(stopwords.words('english'))


# In[666]:


df["Description"] = df["Description"].apply(word_tokenize)


# In[667]:


clean_words = []
for tokenized_description in df["Description"]:
    cleaned_tokens = [token for token in tokenized_description if token not in stop_words]
    clean_words.append(cleaned_tokens)


# In[668]:


df["Description"] = clean_words


# In[669]:


def clean_tokens(tokens):
    cleaned_tokens = []
    for token in tokens:
        cleaned_token = re.sub(r'[^a-zA-Z0-9]', '', token)
        if cleaned_token:
            cleaned_tokens.append(cleaned_token)
    return cleaned_tokens



# In[670]:


df["Description"] = df["Description"].apply(clean_tokens)


# In[671]:


with open('bigram_model.pkl', 'rb') as f:
    bigram = pickle.load(f)


# In[672]:


df['Description'] = df['Description'].apply(lambda tokens: ' '.join(bigram[tokens]))


# In[673]:


with open('vectorizer.pkl', 'rb') as c:
    vectorizer = pickle.load(c)


# In[674]:


transformed_df = vectorizer.transform(df["Description"])


# In[675]:


df_bow = pd.DataFrame(transformed_df.toarray(), columns=vectorizer.get_feature_names_out())


# In[676]:


df.reset_index(drop=True, inplace=True)
df_bow.reset_index(drop=True, inplace=True)


# In[677]:


df_combined = pd.concat([df.drop('Description', axis=1), df_bow], axis=1)


# In[678]:


df_combined = pd.concat([df.drop('Description', axis=1), df_bow], axis=1)


# In[679]:


with open('XGBPipeline.pkl', 'rb') as z:
    bestmodel = pickle.load(z)


# In[680]:


prediction = bestmodel.predict(df_combined)


# In[681]:


price = int(price)


# In[682]:


prediction[0]


# In[683]:


percentage_error = ((price - prediction[0]) / price) * 100


# In[684]:


st.write(f"Predicted Sale Price: ${prediction[0]:.0f}")
st.write(f"Listing vs. Predicted Sale Price Error: ${prediction[0] - price:.0f}")
st.write(f"Percentage Error: {percentage_error:.2f}%")

