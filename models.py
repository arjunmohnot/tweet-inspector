#Imports
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier, LogisticRegression
import pandas as pd, numpy as np, pickle
import matplotlib.pyplot as plt
from sklearn import utils
import re, os, tensorflow as tf, itertools
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils

df = pd.read_excel('c:/users/gagan/desktop/processed.xlsx', encoding = 'utf-8')
df.dropna(inplace = True)
X, y = df.text, df.value
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

#MNB
mnb = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)
ac_mnb = accuracy_score(y_pred, y_test)
print('accuracy %s' % ac_mnb)
pd.DataFrame({'Pred': y_pred, 'Test': y_test}).to_excel('mnb_cls_rp.xlsx')
#print(classification_report(y_test, y_pred))
##filename = 'models/mnb.sav'
##pickle.dump(mnb, open(filename, 'wb'))
##print("Saved MNB to disk")


#LogReg
logreg = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
ac_log_reg = accuracy_score(y_pred, y_test)
print('accuracy %s' % ac_log_reg * 100)
pd.DataFrame({'Pred': y_pred, 'Test': y_test}).to_excel('logreg_cls_rp.xlsx')
#print(classification_report(y_test, y_pred))
##filename = 'models/logistic.sav'
##pickle.dump(logreg, open(filename, 'wb'))
##print("Saved Logistic to disk")

#BOW
train_size = int(len(df) * .7)
train_posts = df['text'][:train_size]
train_tags = df['value'][:train_size]

test_posts = df['text'][train_size:]
test_tags = df['value'][train_size:]

max_words = 1000
tokenizer = text.Tokenizer(num_words=max_words, char_level=False)
tokenizer.fit_on_texts(train_posts) # only fit on train

x_train = tokenizer.texts_to_matrix(train_posts)
x_test = tokenizer.texts_to_matrix(test_posts)

encoder = LabelEncoder()
encoder.fit(train_tags)
#np.save('label_encoder.npy', encoder.classes_)
y_train = encoder.transform(train_tags)
y_test = encoder.transform(test_tags)

num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

batch_size = 32
epochs = 5

# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)

y_pred = model.predict_classes(x_test, batch_size=32, verbose=1)
#ypreds = np.argmax(yp, axis=1)
scores = model.evaluate(x_test, y_test, verbose=1)
#print(list(y_pred.T))
ac_bow = scores[1]
print("Accuracy: %.2f%%" % ac_bow)

#print(history.history['accuracy'])
#print(history.history['loss'])

#pickle.dump(tokenizer, open('models/tokenizer.sav', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
##print('Saved tokenizer to disk')

##model_json = model.to_json()
##with open("models/bow.json", "w") as json_file:
##    json_file.write(model_json)
##model.save_weights("models/bow.h5")
##print("Saved model to disk")

#Plotting
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
#Accuracy score Comparison
plt.bar(np.arange(3), [ac_mnb, ac_log_reg, ac_bow])
plt.title('Accuracy Score Comparison of Algorithms')
plt.xlabel('Algorithm')
plt.ylabel('Accuracy Score')
plt.xticks(np.arange(3), ('Multinomial Naive Bayes', 'Logistic Regression', 'Bag of Words'))
plt.show()
