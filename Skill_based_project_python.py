# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 14:52:51 2025

@author: sneha
"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

################################### Loading dataset ##########################################

data = pd.read_csv("D:\LPU Sem 2\Python for Data Science\python_project\kidney_disease.csv")

################################### Data Cleaning ############################################

#Renaming columns
data.rename(columns={
    'bp': 'blood_pressure',
    'sg': 'specific_gravity',
    'al': 'albumin',
    'su': 'sugar',
    'rbc': 'red_blood_cell',
    'pc': 'pus_cell',
    'pcc': 'pus_cell_clumps',
    'ba': 'bacteria',
    'bgr': 'blood_glucose_random',
    'bu': 'blood_urea',
    'sc': 'serum_creatinine',
    'sod': 'sodium',
    'pot': 'potassium',
    'hemo': 'hemoglobin',
    'pcv': 'packed_cell_volume',
    'wc': 'white_blood_cell_count',
    'rc': 'red_blood_cell_count',
    'htn': 'hypertension',
    'dm': 'diabetes_mellitus',
    'cad': 'coronary_artery_disease',
    'appet': 'appetite',
    'pe': 'pedal_edema',
    'ane': 'anemia'
}, inplace=True)

#Knowing about the data
print(data.info())
print(data.describe())
print(data.isnull().mean()*100)

data['classification'] = data['classification'].str.strip()
data['classification'].value_counts(normalize=True) * 100
data['classification'].value_counts().plot(kind='bar')
#Hence it is a balanced dataset


#Finding NULL values
for i in X.columns:
    print("NULL values in ",i," is ", data[i].isnull().sum()," and its type is ",data[i].dtype)
    
#Filling NULL values
object_type_columns = ['red_blood_cell','pus_cell','pus_cell_clumps','bacteria','hypertension','diabetes_mellitus',
                       'coronary_artery_disease','appetite','pedal_edema','anemia','packed_cell_volume','white_blood_cell_count',
                       'red_blood_cell_count']
float_type_columns = ['age','blood_pressure','specific_gravity','albumin','sugar','blood_glucose_random','blood_urea',
                      'serum_creatinine','sodium','potassium','hemoglobin']

for i in object_type_columns:
    data[i].fillna(data[i].mode()[0], inplace=True)
    
for i in float_type_columns:
    data[i].fillna(data[i].mean(),inplace=True)

#Now finding NULL values
for i in X.columns:
    print("NULL values in ",i," is ", data[i].isnull().sum()," and its type is ",data[i].dtype)
    
######################################### EDA #################################################################################

# Univariate analysis - Histograms
data.hist(bins=15, figsize=(15, 10), color='skyblue', edgecolor='black')
plt.tight_layout()
plt.suptitle("Histograms of Numeric Features", y=1.02)
plt.show()

# Count plots for categorical features
categorical_cols = data.select_dtypes(include='object').columns.tolist()
if categorical_cols:
    for col in categorical_cols:
        plt.figure(figsize=(5,3))
        sns.countplot(data=data, x=col)
        plt.title(f"Count plot of {col}")
        plt.xticks(rotation=45)
        plt.show()
        

# Bivariate - Boxplots for classification
if 'classification' in data.columns:
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
     

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=data, x='classification', y=col)
        plt.title(f"{col} by Classification")
        plt.show()


# Correlation heatmap
numeric_df = data.select_dtypes(include=[np.number])

# Drop columns with all NaNs just in case
numeric_df = numeric_df.dropna(axis=1, how='all')

# Plot correlation heatmap
plt.figure(figsize=(12, 8))
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Heatmap of Numeric Features")
plt.show()


######################################### ANALYSIS #############################################################################
#1)Hemoglobin Analysis 

sns.boxplot(data=data, x='classification', y='hemoglobin')
plt.title("Hemoglobin levels by CKD Classification")
plt.show()

data[['hemoglobin', 'serum_creatinine', 'blood_urea']].corr()

#CKD patients show significantly lower hemoglobin.
#Strong negative correlation with serum creatinine and blood urea

#2)Serum Creatinine Analysis 
sns.boxplot(data=data, x='classification', y='serum_creatinine')
plt.title("Serum Creatinine levels by CKD Classification")
plt.show()

data[['serum_creatinine', 'blood_urea', 'sodium']].corr()

#CKD patients have higher creatinine (a marker of kidney dysfunction).
#Strong positive correlation with blood urea.
#Strong negative correlation with sodium

#3)Packed Cell Volume Analysis

sns.boxplot(data=data, x='classification', y='packed_cell_volume')
plt.title("Packed Cell Volume by CKD Classification")
plt.show()

#CKD patients generally show lower PCV, indicating anemia, a common CKD complication.


#4)Specific Gravity Analysis

sns.boxplot(data=data, x='classification', y='specific_gravity')
plt.title("Specific Gravity Distribution by CKD Classification")
plt.show()

#CKD patients tend to have lower specific gravity, implying reduced urine concentration ability.



#5)Blood Pressure Analysis

sns.boxplot(data=data, x='classification', y='blood_pressure')
plt.title("Blood Pressure Distribution by Classification")
plt.show()

#CKD patients often have higher blood pressure, as hypertension is both a cause and consequence of kidney dysfunction.

#6)Blood Urea vs serum_Creatinine

sns.scatterplot(data=data, x='serum_creatinine', y='blood_urea', hue='classification')
plt.title("Creatinine vs Blood Urea (Colored by Classification)")
plt.show()

#Blood urea and creatinine levels rise together in CKD patients.


#7)Appetite
sns.countplot(data=data, x='appetite', hue='classification')
plt.title("Appetite Levels by CKD Classification")
plt.show()

#Poor appetite is much more frequent among CKD patients.



#8)Presence of Pus Cells clumps (pcc) & Bacteria (ba)

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(data=data, x='pus_cell_clumps', hue='classification', ax=axs[0])
axs[0].set_title("Pus Cell Clumps by Classification")
sns.countplot(data=data, x='bacteria', hue='classification', ax=axs[1])
axs[1].set_title("Bacteria by Classification")
plt.tight_layout()
plt.show()


#These features relate to urinary tract infections, which can complicate CKD.
    
 ###################################LOGISTIC REGRESSION##########################################

# Strip whitespace from column names and values
data.columns = data.columns.str.strip()
data = data.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)

# Replace binary text values with 1/0
binary_map = {
    'yes': 1, 'no': 0,
    'normal': 1, 'abnormal': 0,
    'present': 1, 'notpresent': 0,
    'good': 1, 'poor': 0,
    'ckd': 1, 'notckd': 0
}

for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].map(binary_map)

# Normalize selected features
scaler = MinMaxScaler()
to_normalize = ['blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'blood_glucose_random',
                'blood_urea', 'serum_creatinine', 'sodium', 'potassium', 'hemoglobin']

# Ensure columns exist before normalizing
existing_columns = [col for col in to_normalize if col in data.columns]
data[existing_columns] = scaler.fit_transform(data[existing_columns])

# One-hot encode remaining categorical variables
data = pd.get_dummies(data, drop_first=True)

# Fill missing values with median
data.fillna(data.median(numeric_only=True), inplace=True)

# Tailor features and target
X = data.drop(['id', 'classification_notckd'], axis=1, errors='ignore')
y = data['classification_notckd']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Logistic Regression model
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = log_model.predict(X_test)

# Output results
print("Accuracy:", accuracy_score(y_test, y_pred))


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

####################################SVM##################################################################

 
#train the SVM classifier
svm_model = SVC(kernel='linear')
svm_model.fit(X_train,y_train)    

y_pred = svm_model.predict(X_test)

#evaluate the model
print("accuracy:", accuracy_score(y_test,y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
    
###############################KNN###################################################################

#apply KNN
knn = KNeighborsClassifier(n_neighbors = 22)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print(f"Accuracy of the model: {accuracy*100:.2f}%")  
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show() 
    