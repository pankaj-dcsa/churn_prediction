import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv('churn-bigml-80.csv')
test=pd.read_csv('churn-bigml-20.csv')
print(train.shape)

print(train.head())

print(train.info())

print(train.duplicated().sum())

print(train['Churn'].value_counts())
print(train['State'].value_counts())

print(train.corr())

# Plotting Correlation
plt.figure(figsize=(15,15))
sns.heatmap(train.corr(),annot=True);

sns.pairplot(train,hue="Churn")

# Dropping 'State' as it is irrelevent
train.drop(columns=['State'],inplace=True)
test.drop(columns=['State'],inplace=True)

# One-Hot Encoding
train=pd.get_dummies(train,columns=['International plan','Voice mail plan','Churn'],drop_first=True)
test=pd.get_dummies(test,columns=['International plan','Voice mail plan','Churn'],drop_first=True)

# Input and Output
X_train=train.drop(columns=['Churn_True'])
y_train=train['Churn_True']

X_test=test.drop(columns=['Churn_True'])
y_test=test['Churn_True']

print(X_train.shape)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

print(X_train_scaled)
print(X_test_scaled)

import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model=Sequential()

# Level 1: 3 perceptron, activation function= 'relu' taking 18 input
model.add(Dense(3,activation='relu',input_dim=18))

# Level 2: 3 perceptron, activation function= 'relu'
model.add(Dense(3,activation='relu'))

# Level 3: 1 perceptron, activation function= 'sigmoid'
model.add(Dense(1,activation='sigmoid'))

# Summary of Trainable Parameters
'''
here, 
    Layer 1: 18 x 3 + 3
    Layer 2: 3 x 3 + 3
    Layer 3: 3 x 1 + 1
'''
model.summary()

# Compilation
'''
    loss= 'binary_crossentropy' for binary classification
    optimizer= Adam
'''
model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

# Fitting Model with 100 iteration and splitting to 80%-20% for validation
history=model.fit(X_train_scaled,y_train,epochs=100,validation_split=0.2)

# Trained parameters in Layer 1
model.layers[0].get_weights()

# Trained parameters in Layer 2
model.layers[1].get_weights()

# Predicting
'''
Output passsed through 'Sigmoid' function
Will be gives probability
'''
y_log = model.predict(X_test_scaled)

# converting Provability into 1 and 0
y_pred=np.where(y_log>0.5,1,0)

# Checking Accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

