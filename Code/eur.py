# importing the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Importing dataset
group_names = ['bad', 'good']
dataset = pd.read_csv('../Dataset/insurance.csv')

label = LabelEncoder()
label.fit(dataset.sex.drop_duplicates())
dataset.sex = label.transform(dataset.sex)
label.fit(dataset.smoker.drop_duplicates())
dataset.smoker = label.transform(dataset.smoker)
label.fit(dataset.region.drop_duplicates())
dataset.region = label.transform(dataset.region)
dataset.dtypes
# dataset.replace({"female": 1, "male": 0, "yes": 1, "no": 0}, inplace=True)
print(dataset)

x = dataset.iloc[:, [0, 1, 2, 3, 4, 6]]
y = dataset["charges"]
# x = dataset.iloc[:, [0, 2, 3, 6]].values

# normalizzazione
# sc = StandardScaler()
# x = sc.fit_transform(x)

print("DATASET NORMALIZZATO :")
print(x)

# Finding the optimum number of clusters for k-means classification
wcss = []

for i in range(1, 6):
    kmeansOut = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeansOut.fit(x)
    wcss.append(kmeansOut.inertia_)

# Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 6), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # within cluster sum of squares
plt.show()

# Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters=3)
y_kmeans = kmeans.fit_predict(x)

print("label del y_means", kmeans.labels_)
dataset["cluster"] = kmeans.labels_
print(dataset)
# dataset["cluster"] = dataset["cluster"].map({0: 'Gruppo_2', 1: 'Gruppo_1', 2: 'Gruppo_3'})
print(dataset)
print(kmeans)
# sns.scatterplot(data=dataset, x='age', y='bmi', hue='charges')
# dataset.to_csv(r'C:/Users/forzi/Desktop/Progetto IC 2021 Aprile/Ingegneria_della_Conoscenza_2021/Dataset/prova.csv', index = False, header=True)

# Cluster Plots
plt.title('Clusters of Patients')
sns.scatterplot(data=dataset, x='bmi', y='charges', hue='cluster')
plt.show()

plt.title('BMI and Charges Distribution within Clusters')
sns.scatterplot(data=dataset, x='bmi', y='charges', hue='cluster')
plt.show()

plt.title('Smokers and Charges Distribution within Cluster')
sns.scatterplot(data=dataset, x='smoker', y='charges', hue='cluster')
plt.show()

print(dataset.values)
x_data = dataset[dataset.columns[0:5]].to_numpy()
print("X_DATA.....")
print(x_data)
y_data = np.array(dataset["cluster"])

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle='true', stratify=y_data)


# Function for plotting the results from classification
def plot_results(classifier, x_test, y_test, name):
    pred = classifier.predict(x_test)
    print(classification_report(y_test, pred))
    cm = confusion_matrix(y_test, pred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, fmt='g', cmap="Reds");

    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels');
    ax.set_title(name);
    plt.show()


# Naive Bayes Classifier

classifier1 = GaussianNB()
classifier1.fit(x_train, y_train)
plot_results(classifier1, x_test, y_test, 'Gaussian Naive Bayes')

# KNN Classifier

classifier2 = KNeighborsClassifier()
# defining parameter range
param_grid = {'n_neighbors': [1, 4, 5, 6, 7, 8],
              'leaf_size': [1, 3, 5, 10],
              }
# Hyperparameters Optimization
grid_knn = GridSearchCV(classifier2, param_grid=param_grid)
grid_knn.fit(x_train, y_train)
print("best_params", grid_knn.best_params_)
print("best_estimator", grid_knn.best_estimator_)
plot_results(grid_knn, x_test, y_test, 'K-nearest neighbors')

# Support Vector Machines
# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'sigmoid']}

grid_svm = GridSearchCV(SVC(), param_grid=param_grid, cv=5, refit=True, verbose=False)
grid_svm.fit(x_train, y_train)
print("best_params", grid_svm.best_params_)
print("best_estimator", grid_svm.best_estimator_)
plot_results(grid_svm, x_test, y_test, 'Support Vector Machines')

# questa è la regressione
from sklearn.model_selection import train_test_split as holdout
from sklearn.linear_model import LinearRegression
from sklearn import metrics

x = dataset[dataset.columns[0:6]]
x1 = x.to_numpy()
y = dataset['charges']
x_train, x_test, y_train, y_test = holdout(x1, y, test_size=0.2, random_state=0)
Lin_reg = LinearRegression()
Lin_reg.fit(x_train, y_train)
coeff = pd.DataFrame(Lin_reg.coef_, x.columns, columns=["coefficient"])
print("COEF...")
print(coeff)
print(Lin_reg.intercept_)
print(Lin_reg.coef_)
print(Lin_reg.score(x_test, y_test))

# questa credo la predizione
##Predicting the charges
y_test_pred = Lin_reg.predict(x_test)
##Comparing the actual output values with the predicted values
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
print(df)


def calcinsurance(age, bmi, smoking):
    y = ((age * Lin_reg.coef_[0]) + (bmi * Lin_reg.coef_[2]) + (smoking * Lin_reg.coef_[4]) + Lin_reg.intercept_)
    return y


print("CALCINSURANCE...")
print(calcinsurance(36, 24, 0))
