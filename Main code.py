# Importin libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#--------------------------------------------------------------------------

# Data Collection and Data Processing 

# Reading CSV File
SonarData = pd.read_csv(r'C:\I will prepare my self to destroy the world\Programming\python program\ML project study\1-Sonar Rock VS Mine Predictoin\Data set.csv',header=None)

print(SonarData.head())
# R = Rock
# M = Mine 

#Number rows and columns 
#print(SonarData.shape) # rows = 208 , columns = 61

# Some Statical Defnetion of Data
print(SonarData.describe())

#The count of outputs
print(SonarData[60].value_counts())
# R = 97
#M = 111

print(SonarData.groupby(60).mean())


#Separiting Data and Labels

x = SonarData.drop(columns= 60 , axis= 1)
y = SonarData[60]

#Training and Testing Data
x_train , x_test , y_train , y_test = train_test_split(x , y, test_size=0.1 , stratify= y , random_state=1)

# Model Training ------> Logistic Regression
model = LogisticRegression()

#Trainiing Logistic Regression with Training Data
print(model.fit(x_train , y_train))


#Modle Evaluation 

# Accuracy on Training Data
X_Train_Predict = model.predict(x_train)
X_Train_Accuracy = accuracy_score(X_Train_Predict , y_train)

print(f'Accuracy on Training = {X_Train_Accuracy}')

# Accuracy on Training Data
X_Testing_Predict = model.predict(x_test)
X_Testing_Accuracy = accuracy_score(X_Testing_Predict , y_test)

print(f'Accuracy on Training = {X_Testing_Accuracy}')

#Macking  a Predictive System 
print('\n', '-'*100,'\n')

temp_list = []
for i in range(60):
    value = float(input(f"Enter value {i+1}: "))
    temp_list.append(value)
input_Data = tuple(temp_list)

#changing input_Data to a numpy array
input_Data_as_Array = np.asarray(input_Data)

#Reshape the array
Reshaped_Data = input_Data_as_Array.reshape(1,-1)

predict_from_model = model.predict(Reshaped_Data)
#print(f'The prediction for the Data you want is {predict_from_model}')

if predict_from_model[0] == 'R':
    print('That Data is a Data for a Rock\n')
elif predict_from_model[0] == 'M':
    print('Taht Data is a Data for a Mine\n')

# now you have  a full Model to predict if the object is a mine of a Tock