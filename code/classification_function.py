import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

scoring = {'accuracy': make_scorer(accuracy_score),
           'precision': make_scorer(precision_score, average='macro', zero_division=0),
           'recall': make_scorer(recall_score, average='macro', zero_division=0)}


def KNNClassification(training, target):
    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size=0.3, random_state=0)
    knn = KNeighborsClassifier(metric='manhattan', n_neighbors=13, weights='distance')
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    print("Accuracy knn:", metrics.accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return knn


def GaussianNBClassification(training, target):
    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size=0.3, random_state=0)
    gau = GaussianNB(var_smoothing= 0.4329)
    gau.fit(x_train, y_train)
    y_pred_gau = gau.predict(x_test)
    print("Accuracy gau :", metrics.accuracy_score(y_test, y_pred_gau))
    print(classification_report(y_test, y_pred_gau))


def RandomForestClassifierClassification(training, target):
    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size=0.3, random_state=0)
    rf = RandomForestClassifier(max_depth= 15, max_features= 3, min_samples_leaf= 3, min_samples_split= 3, n_estimators= 1300, criterion='entropy')
    rf.fit(x_train, y_train)
    y_pred_rf = rf.predict(x_test)
    print("Accuracy rf:", metrics.accuracy_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf))

    return rf


def evaluate_with_crossvalidate(knn_model, x_train, y_train, cv):
    knn = cross_validate(knn_model, x_train, y_train, cv=cv, scoring=scoring)
    models_scores_table = pd.DataFrame(
        {'KNearestNeighbor': [knn['test_accuracy'].mean(),
                              knn['test_precision'].mean(),
                              knn['test_recall'].mean()]},
        index=['Accuracy', 'Precision', 'Recall'])

    models_scores_table.to_csv("dataset/models_scores_table.csv", index=False)


def evaluate_with_crossvalidaterf(knn_model, x_train, y_train, cv):
    knn = cross_validate(knn_model, x_train, y_train, cv=cv, scoring=scoring)
    models_scores_table = pd.DataFrame(
        {'Random Forest': [knn['test_accuracy'].mean(),
                              knn['test_precision'].mean(),
                              knn['test_recall'].mean()]},
        index=['Accuracy', 'Precision', 'Recall'])

    models_scores_table.to_csv("dataset/models_scores_table_randomForest.csv", index=False)