# Import packages and librairies
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# Read the csv file
df =pd.read_csv("/users/donaldtrump/desktop/csv/titanic.csv")
print("first 5 rows")
print(df.head())
print(df.describe())
print(df.info())
df.shape
# Explore our data
print(df['Survived'].value_counts())
print(df['Sex'].value_counts())
print(df['Pclass'].value_counts())
print(df.isnull().sum())
# Cleaning and wrangling
df['Age'].fillna(df['Age'].median(),inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace=True)

df['Sex']=df['Sex'].map({'male':0 ,'female':1})
df['Familysize']=df['SibSp']+df['Parch']
df['Isalone']=df['Familysize'].apply(lambda x:1 if x==0 else 0)
df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)
df =pd.get_dummies(df,columns=['Embarked','Pclass'],drop_first=True)
print(df.head())
print(df.info())
# Survival counts
sns.set(style='whitegrid')
sns.countplot(x='Survived',data=df)
plt.title("Survival Count")
plt.show()
# Survival by sex
sns.countplot(x='Sex',hue='Survived',data=df)
plt.xlabel("Sex")
plt.ylabel("Survived")
plt.title("Survival By Sex")
plt.show()
# Survival By Class
sns.countplot(x='Survived', hue='Pclass_2', data=df)
plt.title("Survival by Pclass (2nd Class)")
plt.show()
plt.figure(figsize=(8,5))
sns.histplot(data=df,x='Age',hue='Survived',bins=30,kde=True,multiple='stack')
plt.title("Age Distribution by survival status")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
# Predictive_Modeling

 # Logistic regression as a predictive model
     # Split the data
x=df.drop('Survived',axis=1)
y=df['Survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=65)
     # Train the model
model =LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
     # Evaluate the model
y_pred =model.predict(x_test)
accuracy1=accuracy_score(y_test,y_pred)
print(accuracy1)
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt='d',cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

 # Randomforest model
     # Split the data
x1=df.drop('Survived',axis=1)
y=df['Survived']
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y,test_size=0.2,random_state=42)
     # Train the model
rf=RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x1_train,y1_train)
y_pred1=rf.predict(x1_test)
     # Evaluate the model
accuracy2=accuracy_score(y1_test,y_pred1)
print(accuracy2)
print("\nConfusion Matrix:\n", confusion_matrix(y1_test, y_pred1))
import matplotlib.pyplot as plt
import pandas as pd

# Get feature importances from the trained model
importances = rf.feature_importances_

# Create a DataFrame for better visualization
feat_importances = pd.Series(importances, index=x1.columns)
feat_importances = feat_importances.sort_values(ascending=False)


plt.figure(figsize=(10,6))
feat_importances.plot(kind='bar')
plt.title('Feature Importances from Random Forest')
plt.ylabel('Importance')
plt.show()