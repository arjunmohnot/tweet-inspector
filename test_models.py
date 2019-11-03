from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score as acc, confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier, LogisticRegression
import pandas as pd, numpy as np, pickle
from sklearn import utils
import re, clean, tensorflow as tf
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils
from keras.models import model_from_json
import os
from settings import APP_STATIC


def get_clean_data(df):

    df = clean.clean(df)
    xa_test, y_test = np.array(df.text.tolist()), np.array(df.value.tolist())
    return (xa_test, y_test)

def test_models(df, valflg):

    model_list = ['mnb.sav', 'logistic.sav', 'bow.json']
    x_test, y_test = get_clean_data(df)
    c = 2
    for i in model_list:

        if '.json' not in i:
            filename = f'models/{i}'
            loaded_model = pickle.load(open(filename, 'rb'))
            result = loaded_model.predict(x_test)#, y_test)
            print(i, ':', acc(y_test, result))

        else:
            handle = open('models/tokenizer.sav', 'rb')
            tokenizer = pickle.load(handle)
            x_test = tokenizer.texts_to_matrix(x_test)
            json_file = open(f'models/{i}', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            loaded_model = model_from_json(loaded_model_json)
            # load weights into new model
            loaded_model.load_weights("models/bow.h5")
            encoder = LabelEncoder()
            encoder.classes_ = np.load('models/label_encoder.npy')
            y_test = encoder.transform(y_test)
            y_test = utils.to_categorical(y_test, 2)
            # evaluate loaded model on test data
            loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            score = loaded_model.evaluate(x_test, y_test, verbose=0)
            print(i, ": %s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
            result = loaded_model.predict_classes(x_test)
            #result = list(result.T[0])
            
        df.insert(c, i, 0)
        df[i] = list(result)
        c = c+1
        
    df.to_excel(os.path.join(APP_STATIC, f'pred1.xlsx'), encoding = 'utf-8')
    return df

def test_main(json_inp):

    txt = list()
    value = list()
    
    for i in range(len(json_inp)):
        txt.append(json_inp[i].get('text', ''))
        value.append(json_inp[i].get('value', 0))

    df_init = pd.DataFrame({'text': txt, 'value': value}, index = list(range(len(txt))))
    valflg = 1
    if 'value' not in df_init.columns:
        valflg = 0
    #df_init = get_test_set(file)
    #df_init = pd.DataFrame({'text': ['चूतिया हो तुम भोसड़ी के', 'चुटिया लम्बी है तुम्हारी', 'सुना है लोग उसे आँख भर कर देखते हैं', 'तो थोड़े दिन उसके शहर में होकर देखते हैं'],
                            #'value': [1, 0, 0, 0]}, index = [0, 1, 2, 3])
    print(df_init)
    df = test_models(df_init, valflg)
    cols = ['text', 'mnb.sav', 'logistic.sav', 'bow.json']
        
    json_dict = dict()
    for i in df.index:
        d = [(cols[j], str(df.loc[i, cols[j]])) for j in range(len(cols))]
        if valflg: d.append(('value', str(df.loc[i, 'value'])))
        json_dict[i] = dict(d)

    return json_dict


#if __name__ == '__main__':

    #df_init = get_test_set('C:/Users/gagan/Desktop/abuse/TweetScraper-master/TweetScraper/spiders/Data/tweets1.xlsx')
    #df_init = pd.DataFrame({'text': ['चूतिया हो तुम भोसड़ी के', 'चुटिया लम्बी है तुम्हारी', 'सुना है लोग उसे आँख भर कर देखते हैं', 'तो थोड़े दिन उसके शहर में होकर देखते हैं'],
                            #'value': [1, 0, 0, 0]}, index = [0, 1, 2, 3])
    #df_init = pd.read_excel('c:/users/gagan/desktop/processed.xlsx').sample(frac=1)
    #test_models(df_init)
