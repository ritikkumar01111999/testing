import pandas as pd
import pandas as pd
import numpy as np
import csv 

import warnings

import pickle
import time

import re

import nltk
from nltk.tokenize import ToktokTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation


from sklearn import metrics
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import hamming_loss
from sklearn.model_selection import train_test_split
import joblib

from flask import Flask,request, jsonify
from flask import app
warnings.filterwarnings("ignore")



df = pd.read_csv('data.csv')
# print(df.head())
# function to clean data
nltk.download('stopwords')
nltk.download('wordnet')
stop_word = stopwords.words('english')
stop = stopwords.words('english')


def cleanig_data(dataset):


    ''' function to clean text data . It will remove all special char , HTML tags and null values'''
    # removing 'Unnamed: 0' , which is present in dataset bcz of no use.
    dataset.drop(columns=['Unnamed: 0'], inplace=True)
  # removing nan values
    dataset = dataset.dropna()
  # changing all description text into lower form
    dataset['description'] = dataset['description'].str.lower()
  # changing all tags text into lower form
    dataset['tags'] = dataset['tags'].str.lower()

    def cleanHtml(sentence):
    #   ''' It cleans all HTML tags '''
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, ' ', str(sentence))
        return cleantext
    def cleanPunc(sentence):

        '''function to clean the word of any punctuation or special characters'''
        cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
        cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
        cleaned = cleaned.strip()
        cleaned = cleaned.replace("\n"," ")
        return cleaned
    def keepAlpha(sentence):
        ''' only keeping words with no special character'''
        alpha_sent = ""
        for word in sentence.split():
            alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
            alpha_sent += alpha_word
            alpha_sent += " "
        alpha_sent = alpha_sent.strip()
        return alpha_sent
  
    token=ToktokTokenizer()
    lemma=WordNetLemmatizer()

    def lemitizeWords(text):
        ''' function to apply lemmatizer on text'''
        words=token.tokenize(text)
        listLemma=[]
        for w in words:
            x=lemma.lemmatize(w, pos="v")
            listLemma.append(x)
        return ' '.join(map(str, listLemma))
  # applying all defined fuction on dataset columns
    dataset['description'] = dataset['description'].apply(cleanHtml)
    dataset['description'] = dataset['description'].apply(cleanPunc)
    dataset['description'] = dataset['description'].apply(keepAlpha)  
    dataset['description'] = dataset['description'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    dataset['description'] = dataset['description'].apply(lambda x: lemitizeWords(x))
  # returning cleaned dataset
    return dataset

# calling function on originsl dataset
data_result = cleanig_data(df)
# print(data_result.head())

# Tag Processing
# split tags in space
data_result['new_tags'] = data_result["tags"].apply(lambda x: x.split())
# add all tags
all_tags = [item for sublist in data_result['new_tags'].values for item in sublist]
# copying cleaned dataset into cleaned_data dataframe
cleaned_data = data_result.copy()
# printing some part of dataset
# print(cleaned_data.head())

# print(cleaned_data.shape)


# train - test Split
# Defining X and y
X = cleaned_data['description']
y = cleaned_data['new_tags']

# encoding MultiLabelBinarizer for dependent lables
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html
mulitlabel_binarizer = MultiLabelBinarizer()
y_bin = mulitlabel_binarizer.fit_transform(y)
classes = mulitlabel_binarizer.classes_
# print(classes)

# encoding independent labels or descriptions
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
vectorizer = TfidfVectorizer(analyzer = 'word',
                                       min_df=0.0,
                                       max_df = 1.0,
                                       strip_accents = None,
                                       encoding = 'utf-8', 
                                       preprocessor=None,
                                       max_features=1000)

multilabel_x_data = vectorizer.fit_transform(X)
# splitting encoded x and y into X_train ,X_test,y_test,y_train with train-test ratio of 70:30
X_train, X_test, y_train, y_test = train_test_split(multilabel_x_data, y_bin, test_size=0.3,random_state=123)
# printing shape of X-train , X-test ,y-train,y-test
# print("Dimensions of train data X:",X_train.shape, "Y :",y_train.shape)
# print("Dimensions of test data X:",X_test.shape,"Y:",y_test.shape)

# now taining data with Onevsrest classifier technique using logistic regression algorithm
classifier = OneVsRestClassifier(SGDClassifier(loss='log', alpha=0.00001, penalty='l1'))
# fitting X_train,y_train on classifier defined above
classifier.fit(X_train,y_train )
# predicting y value by giving X_test as input
predictions = classifier.predict (X_test)
# predicting y value by giving X_train as input
predictions_train = classifier.predict (X_train)

# assigning y_test to y_true
y_true = y_test
# assigning predicted y value of X_test to y_logits
y_logits = predictions

#  '''it is computing fraction of equality of two array'''
def result_check(x, y):
  count = 0
  for i in range(len(x)):
    if x[i] == y[i]:
      count += 1
  result = count / len(x)
  return result
# https://stackoverflow.com/questions/46799261/how-to-create-an-exact-match-eval-metric-op-for-tensorflow
def subset_accuracy(y_true,y_predict,threshold):
  ''' it is computing accuracy of model on the basis of , equality fraction between true values and prediction values '''
  count = 0
  for j in range(len(y_true)):
    if result_check(y_true[j] , y_predict[j]) >= threshold:
      count = count + 1
  accuracy = count / len(y_true)
  return accuracy
# giving some set of equality fraction and findin accuracy
threshold_list = [1,.9,.8,.7,.6,.5]
for i in threshold_list:
  print(f'for threshold {i} accuracy is = ',subset_accuracy(y_true,y_logits,threshold=i))

# printing overall train accuracy of model 
print("train Accuracy :",metrics.accuracy_score(y_train, predictions_train))
# printing overall test accuracy of model
print("test Accuracy :",metrics.accuracy_score(y_test, predictions))
# printing hamming loss of  test data
print("Hamming loss ",metrics.hamming_loss(y_test,predictions))

# printing micro precision of test data
precision = precision_score(y_test, predictions, average='micro')
# printing micro recall of test data
recall = recall_score(y_test, predictions, average='micro')
# printing micro f1-score of test data
f1 = f1_score(y_test, predictions, average='micro') 
print("Micro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
# printing macro precision of test data
precision = precision_score(y_test, predictions, average='macro')
# printing macro recall of test data
recall = recall_score(y_test, predictions, average='macro')
# printing macro f1-score of test data
f1 = f1_score(y_test, predictions, average='macro') 
print("Macro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

# saving the model using joblib
joblib.dump(classifier , 'model.pkl')
joblib.dump(vectorizer , 'vectorizer.pkl')

# to map index with class name
def print_class(arr):
  cls_list = []
  for index, value in enumerate(arr):
    if value == 1:
      cls_list.append(classes[index])
  return cls_list
   
# loading saved model
# saving vectorizer as vect
vect = joblib.load('vectorizer.pkl')
# saving classifier as model
model = joblib.load('model.pkl')

def simlr(testing_result,true):
    ''' it will compute how similar two lists are'''
    count = 0
    for i in true:
        if i in testing_result:
            count = count+1
    return count/len(true)
app=Flask(__name__)
@app.route('/')
def main():
    # loading saved model
    # saving vectorizer as vect
    vect = joblib.load('vectorizer.pkl')
    # saving classifier as model
    model = joblib.load('model.pkl')

    testing=request.form.get('testing')
    v = vect.transform([testing])
    p = model.predict(v)
    testing_result=print_class(p[0])
    # true = df['tags'][1].split()
    # sm = simlr(testing_result,true)
    return jsonify(testing , testing_result)



if __name__ == '__main__':
    app.run()



# testing = df['description'][1]
# # encodig input text
# v = vect.transform([testing])
# # predicting model
# p = model.predict(v)
#   # mappin result with classes
# testing_result=print_class(p[0])

#   # retriving real tags from dataset and splitting them bcz they string
# true = df['tags'][1].split()

# # computing similarity between real tags and predicted tags
# sm = simlr(testing_result,true)

# # print(f'input=',{testing},'true_value=',{true},'predicted_value='{testing_result},'similarity',sm)
# print(testing,testing_result,true,sm)