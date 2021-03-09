import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def validation(test, prediction):
    from sklearn.metrics import classification_report
    print(classification_report(test, prediction))
    
    print("\n\n")


def confusionMatrix(test, prediction, classes_name,name):
    # Plot non-normalized confusion matrix
    matrix = metrics.confusion_matrix(test, prediction)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=classes_name,
                yticklabels=classes_name,
                annot=True,
                fmt='d')
    plt.title(name)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

