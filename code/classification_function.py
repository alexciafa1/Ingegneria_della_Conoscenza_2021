from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


def knn_classification(training, target):

    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size=0.3, random_state=0)

    knn = KNeighborsClassifier(metric='manhattan', n_neighbors=13, weights='distance')

    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)

    # print("Accuracy knn:", metrics.accuracy_score(y_test, y_pred))

    # print(classification_report(y_test, y_pred))

    return knn


def gaussian_nb_classification(training, target):

    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size=0.3, random_state=0)

    gau = GaussianNB(var_smoothing=0.4329)

    gau.fit(x_train, y_train)

    y_pred_gau = gau.predict(x_test)

    # print("Accuracy gau :", metrics.accuracy_score(y_test, y_pred_gau))

    # print(classification_report(y_test, y_pred_gau))


def random_forest_classification(training, target):

    x_train, x_test, y_train, y_test = train_test_split(training, target, test_size=0.3, random_state=0)

    rf = RandomForestClassifier(max_depth=15, max_features=3, min_samples_leaf=3, min_samples_split=3,
                                n_estimators=1300, criterion='entropy')

    rf.fit(x_train, y_train)

    y_pred_rf = rf.predict(x_test)

    # print("Accuracy rf:", metrics.accuracy_score(y_test, y_pred_rf))

    # print(classification_report(y_test, y_pred_rf))

    return rf
