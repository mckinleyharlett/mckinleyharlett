#!/usr/bin/env python
# coding: utf-8

# # Neural Network Classifier Exercises

# In[37]:


# Packages 
import random
import json
import pandas as pd
import jsonlines
import sys
import unicodedata
import nltk
from nltk.corpus import stopwords
## nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[2]:


# Loading the data
data = []
 
with jsonlines.open('categorized-comments.jsonl') as reader:
    for obj in reader.iter(type=dict, skip_invalid=True):
        data.append(obj)


# In[3]:


df = pd.DataFrame(data=data)
df.head()


# ### Cleaning the data

# In[10]:


# Getting a smaller sample
df1 = df.sample(n=10000)


# In[11]:


# Now we are going to be convering all text in the rows to lowercase
df1['lowertext'] = df1['txt'].apply(lambda x: x.lower())


# In[12]:


# Removing any unnecessary punctuation 
punctuation = dict.fromkeys(i for i in range(sys.maxunicode)
                            if unicodedata.category(chr(i)).startswith('P')) 


# In[13]:


# Applying the punctuation step to our text field
df1['lowertext'] = [string.translate(punctuation) for string in df1['lowertext']]


# In[14]:


# Tokenizing the words from each row
tokenized_words = df1.apply(lambda row: nltk.word_tokenize(row['lowertext']), axis=1)


# In[43]:


# I need to make the category column into a numerical
df1['cat'] = pd.Categorical(df1['cat'])
df1['code'] = df1['cat'].cat.codes


# # 1. Neural Network Classifier with Scikit

# In[44]:


# Generating TFIDF values 
tfidfconverter = TfidfVectorizer(max_features = 50000, stop_words = stopwords.words('english'))

X = tfidfconverter.fit_transform(df1['lowertext']).toarray()


# In[45]:


# Splitting our data into training and testing datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, df1['code'].values, test_size=0.25, random_state=0)


# In[46]:


# Fitting neural network model using our training data
import warnings
warnings.filterwarnings('ignore')

classifier = MLPClassifier(hidden_layer_sizes = (30,30,30), max_iter = 500, random_state=0)
classifier.fit(X_train, Y_train)


# In[47]:


# Predicting the comment categories by comparing the test category
Y_predict = classifier.predict(X_test)
Y_predict


# In[55]:


# Printing results
print('Accuracy: \n {} \n'.format(accuracy_score(Y_test, Y_predict)))
print('Classification Report: \n {} \n'.format(classification_report(Y_test, Y_predict)))
print('Confusion Matrix: \n {} \n'.format(confusion_matrix(Y_test, Y_predict)))


# # 2. Neural Network Classifier with Keras

# In[49]:


# Packages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras import layers


# In[50]:


# Creating a function so then it'll be faster and easier for me to understand
# I thought since I did the first one, i'd try making a function! 
def train_test_model(X_train, Y_train, X_test):
    
    # Setting up our model 
    model = Sequential()
    # Adding 3 layers with neuron numbers and activation functions
    model.add(Dense(300, activation = 'relu', 
                    input_dim = X_train.shape[1]))
    model.add(Dense(700, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    
    # Compile the model 
    model.compile(
                loss = 'sparse_categorical_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy'])
    
    # Fitting the model to our training datasets
    model.fit(X_train, Y_train, 
              batch_size = 500, 
              epochs = 10, 
              verbose = 1)
    
    # Predicting categories from the comments
    y_pred = np.argmax(model.predict(X_test), axis = -1)
    
    return y_pred


# In[51]:


# Fit model to our training data!
y_pred = train_test_model(X_train, Y_train, X_test)


# In[56]:


print('Accuracy: \n {} \n'.format(accuracy_score(Y_test, y_pred)))
print('Confustion Matrix: \n {} \n'.format(confusion_matrix(Y_test, y_pred)))
print('Classification Report: \n {} \n'.format(classification_report(Y_test, y_pred)))


# In[ ]:




