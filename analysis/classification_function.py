import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def knn_classification(training, target):

    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size=0.3, random_state=0)
    '''
    knn = KNeighborsClassifier()
    parameters_knn = {
        'n_neighbors': (1, 10, 13),
        'weights': ('uniform', 'distance'),
        'metric': ('minkowski', 'manhattan')}

    # GridSearch
    grid_search_knn = GridSearchCV(
        estimator=knn,
        param_grid=parameters_knn,
        scoring='accuracy',
        n_jobs=-1,
        cv=5
    )

    knn_1 = grid_search_knn.fit(x_train, y_train)
    y_pred_knn1 = knn_1.predict(x_test)
    print("Best params", grid_search_knn.best_params_)
    '''
    knn = KNeighborsClassifier(metric='manhattan', n_neighbors=13, weights='distance')

    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)

    # print("Accuracy knn:", metrics.accuracy_score(y_test, y_pred))

    # print(classification_report(y_test, y_pred))

    return knn


def gaussian_nb_classification(training, target):

    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size=0.3, random_state=0)
    '''
    gau = GaussianNB()
    parameters_gau = {'var_smoothing': np.logspace(0, -9, num=100)}

    # GridSearch
    grid_search_gau = GridSearchCV(
        estimator=gau,
        param_grid=parameters_gau,
        cv=10,  # use any cross validation technique
        verbose=1,
        scoring='accuracy'
    )

    gau_1 = grid_search_gau.fit(x_train, y_train)
    y_pred_gau1 = gau_1.predict(x_test)
    print("Best params", grid_search_gau.best_params_)
    '''

    gau = GaussianNB(var_smoothing=0.4329)

    gau.fit(x_train, y_train)

    y_pred_gau = gau.predict(x_test)

    # print("Accuracy gau :", metrics.accuracy_score(y_test, y_pred_gau))

    # print(classification_report(y_test, y_pred_gau))


def random_forest_classification(training, target):

    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size=0.3, random_state=0)
    '''
    rf = RandomForestClassifier()

    parameters_rf = {
        'n_estimators': (100, 500, 1300),
        'max_depth': (15, 25),
        'min_samples_split': (3, 15),
        'min_samples_leaf': (1, 3)

    }

    grid_rf = GridSearchCV(rf, parameters_rf, cv=10, verbose=1,
                         scoring='accuracy')
    rf1 = grid_rf.fit(x_train, y_train)
    y_pred_rf1 = rf1.predict(x_test)
    print("Best params", grid_rf.best_params_)
    '''

    rf = RandomForestClassifier(max_depth=15, max_features=3, min_samples_leaf=3, min_samples_split=3,
                                n_estimators=1300, criterion='entropy')

    rf.fit(x_train, y_train)

    y_pred_rf = rf.predict(x_test)

    # print("Accuracy rf:", metrics.accuracy_score(y_test, y_pred_rf))

    # print(classification_report(y_test, y_pred_rf))

    return rf
