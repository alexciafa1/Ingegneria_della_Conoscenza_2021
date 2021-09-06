import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import model_selection, metrics
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate

def merge_with_rating(anime):
    rating = pd.read_csv('dataset/rating.csv')

    MRPU = rating.groupby(['anime_id']).mean().reset_index()

    MRPU['mean_rating'] = MRPU['rating']

    MRPU.drop(['user_id', 'rating'], axis=1, inplace=True)

    MRPU.to_csv("dataset/mru.csv", index=False)

    mergedata = pd.merge(anime, MRPU, on=['anime_id'])

    mergedata = mergedata[mergedata.mean_rating >= 0]

    return mergedata
'''
user = pd.merge(user,MRPU,on=['user_id','user_id'])

user = user.drop(user[user.rating < user.mean_rating].index)

user["user_id"].unique()

mergedata = pd.merge(anime,user,on=['anime_id','anime_id'])

mergedata= mergedata[mergedata.user_id <= 20000]
'''
