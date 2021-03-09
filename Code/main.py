import classifier as classifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import classifiers
import metrics
warnings.filterwarnings('ignore')


df = pd.read_csv('C:/Users/Regina/Desktop/Progetto IX/dataset/Heart_Disease_Prediction.csv')
sns.countplot(x='Heart Disease', data=df)
group_names=['Absence', 'Presence']

categorical_features=["Sex","Chest pain type","FBS over 120","EKG results","Exercise angina","Slope of ST","Number of vessels fluro","Thallium"]

df[categorical_features]=df[categorical_features].astype("category")


# dataset visualization
print('Number of rows in the dataset: ', df.shape[0])
print('Number of columns in the dataset: ', df.shape[1])
#print(df[categorical_features])
print(df.dtypes)



# Splitting the dataset into training and testing sets.
x = df.iloc[:, :-2]
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 0, test_size = 0.35)

# Using standard scaler as a standardization technique.
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

print("\n\n")
# Looking for optimal number of nearest neighbours.
import math
print("Optimal number of nearest neighbours:",math.sqrt(len(y_test)))


from sklearn.decomposition import PCA

pca = PCA()
X_pca = pca.fit_transform(x_train)
variance = pca.explained_variance_ratio_  # calculate variance ratios
var = np.cumsum(np.round(variance, decimals=3) * 100)
print("\n\n")
print(var)

plt.figure(figsize=(7, 6))
plt.ylabel('% Variance Explained')
plt.xlabel('# of Features')
plt.title('PCA Analysis')
plt.plot(var, 'ro-')
plt.grid()
print("\n\n")


#KNN Model.
classifier1 = classifiers.knnClassifier(x_train, y_train)
prediction = classifier1.predict(x_test)
print(classifier1)
metrics.validation(y_test, prediction)
metrics.confusionMatrix(y_test, prediction, name =' Confusion Matrix KNN')
print("\n\n")

#Naive Bayes
classifier2= classifiers.bayesianClassifier(x_train,y_train)
prediction = classifier2.predict(x_test)
print(classifier2)
metrics.validation(y_test, prediction)
metrics.confusionMatrix(y_test, prediction, name= 'Confusion Matrix Bayes')
print("\n\n")

#SVM

classifier3= classifiers.SVMClassifier(x_train,y_train)
prediction = classifier3.predict(x_test)
print(classifier3)
metrics.validation(y_test, prediction)
metrics.confusionMatrix(y_test, prediction, name='Confusion Matrix SVM')
print("\n\n")

#RandomTree Classifier
classifier4 = classifiers.randomForestClassifier(x_train,y_train)
prediction = classifier4.predict(x_test)
print(classifier4)
metrics.validation(y_test, prediction)
metrics.confusionMatrix(y_test, prediction, name='Confusion Matrix RFC')
print("\n\n")





importances = classifier4.feature_importances_ # importanza delle singole feature

std = np.std([tree.feature_importances_ for tree in classifier4.estimators_], axis=0)
indices = np.argsort(importances)[::-1]


# build plot
std = np.std([tree.feature_importances_ for tree in classifier4.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
plt.figure()
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]), indices)
plt.xlim([-1, x_train.shape[1]])
plt.show()
