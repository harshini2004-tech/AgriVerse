import numpy as np
import pandas as pd

# Use raw string or forward slashes
# data = pd.read_csv(r"C:\Users\Skanda\OneDrive\Desktop\Major Project\Project_Code\AggriAssist_Fertilizer_prediction\Fertilizer Prediction.csv")
data = pd.read_csv("C:/Users/Skanda/OneDrive/Desktop/Major Project/Project_Code/AggriAssist_Fertilizer_prediction/Fertilizer Prediction.csv")

# print(data.head())

soil_dict={
    'Loamy':1,
    'Sandy':2,
    'Clayey':3,
    'Black':4,
    'Red':5
}

crop_dict={
    'Sugarcane':1,
    'Cotton':2,
    'Millets':3,
    'Paddy':4,
    'Pulses':5,
    'Wheat':6,
    'Tobacco':7,
    'Barley':8,
    'Oil seeds':9,
    'Ground Nuts':10,
    'Maize':11
    
}

data['Soil_Num']=data['Soil Type'].map(soil_dict)
data['Crop_Num']=data['Crop Type'].map(crop_dict)

data=data.drop(['Soil Type','Crop Type'],axis=1)
print(data.head())

X=data.drop(['Fertilizer Name'],axis=1)
Y=data['Fertilizer Name']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# create instances of all models
models = {
    # 'Logistic Regression': LogisticRegression(),
    # 'Naive Bayes': GaussianNB(),
    # 'Support Vector Machine': SVC(),
    # 'K-Nearest Neighbors': KNeighborsClassifier(),
    # 'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    # 'Bagging': BaggingClassifier(),
    # 'AdaBoost': AdaBoostClassifier(),
    # 'Gradient Boosting': GradientBoostingClassifier(),
    # 'Extra Trees': ExtraTreeClassifier(),
}

for name,md in models.items():
    md.fit(X_train,Y_train)
    ypred=md.predict(X_test)
    
    # print(f"the Accuracy of {name} is ",accuracy_score(Y_test,ypred))


classifier=RandomForestClassifier()
classifier.fit(X_train,Y_train)
ypred=classifier.predict(X_test)

def recommendation(Temparature,Humidity,Moisture,Nitrogen,Potassium,Phosphorous,Soil_Num,Crop_Num):
    features = np.array([[Temparature,Humidity,Moisture,Nitrogen,Potassium,Phosphorous,Soil_Num,Crop_Num]])
    prediction = classifier.predict(features).reshape(1,-1)
    
    return prediction[0] 

Temparature=2
Humidity=59
Moisture=3
Nitrogen=12
Potassium=0
Phosphorous=3
Soil_Num=2
Crop_Num=11
predict=recommendation(Temparature,Humidity,Moisture,Nitrogen,Potassium,Phosphorous,Soil_Num,Crop_Num)
print(predict[0])



import pickle
pickle_out = open("Fertclassifier_random2.pkl","wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()