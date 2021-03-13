from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def KNNclassifier (x_value,y_value):
    print('building KNNclassifier...')
    classifier = KNeighborsClassifier(n_neighbors=4, algorithm='auto', p=2, metric='minkowski', leaf_size=5)
    classifier.fit(x_value, y_value)
    return classifier


def BayesianClassifier (x_value,y_value):
    print('building BayesianClassifier...')
    classifier = GaussianNB()
    classifier.fit(x_value, y_value)
    return classifier

