import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import itertools

# sklearn.
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

# Borro columnas según diccionario
def get_x_data(new_vector_features, cols):
    if 'label' in new_vector_features.columns:
        x_data = new_vector_features.drop('label', axis = 1)
    else:
        x_data = new_vector_features
    for keys in cols.keys():
        if cols[keys] is False:
            x_data = x_data.drop(keys, axis = 1)
    return x_data, new_vector_features.label

'''
Creo un vector con el formato del envío final
'''
def makeSubmission(y_pred, y_labels_to_predict, threshold = 0.5):
    to_send = pd.Series(y_pred[:,1]>threshold).astype(int)
    to_send = pd.DataFrame(to_send)
    to_send = to_send.rename(columns={ 0 : 'label'})
    to_send.index = y_labels_to_predict['person']
    to_submit2 = y_labels_to_predict.drop('label', axis= 1)
    to_submit2 = to_submit2.join(to_send,on = 'person', how = 'right')
    to_submit2['label'].unique()
    to_submit2 = to_submit2.set_index('person')
    return to_submit2

def plot_precision_and_recall(precision, recall, threshold):
    plt.clf()
    plt.plot(threshold, precision[:-1], "r-", label="precision", linewidth=5)
    plt.plot(threshold, recall[:-1], "b", label="recall", linewidth=5)
    plt.xlabel("threshold", fontsize=19)
    plt.legend(loc="upper right", fontsize=19)
    plt.ylim([0, 1])
    plt.show()
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.clf()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def runRandomForests(X_train, y_train, X_test, y_test, n_estimators = 100, criterion = 'gini', min_samples_split=4):
    # Creo el objeto
    random_forest = RandomForestClassifier(n_estimators = n_estimators,
                                           min_samples_split = min_samples_split,
                                           criterion = criterion,
                                           random_state = 123)
    # Ajusto
    random_forest.fit(X_train, y_train)

    # Predigo
    Y_prediction = random_forest.predict(X_test)

    # Calculate the absolute errors
    errors = abs(Y_prediction - y_test)

    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    
    #random_forest.score(X_test, Y_prediction)
    acc_random_forest = round(random_forest.score(X_test, Y_prediction) * 100, 2)
    print(round(acc_random_forest,2,), "%")
    return random_forest

def runXGBoost(X_train, y_train, X_test, y_test, max_depth = 5, learning_rate = 0.1, n_estimators = 100,
              n_jobs = 2, min_child_weight=1):  #, criterion = 'gini', min_samples_split=4):
    # Creo el objeto
    xgb = XGBClassifier(n_estimators = n_estimators,
                        max_depth = max_depth, learning_rate = learning_rate,
                        random_state = 123, n_jobs = n_jobs,objective= 'binary:logistic',
                        gamma=0, subsample=0.8, colsample_bytree=0.8, min_child_weight=min_child_weight,
                        scale_pos_weight=1)
    # Ajusto
    xgb.fit(X_train, y_train)

    # Predigo
    Y_prediction = xgb.predict(X_test)

    # Calculate the absolute errors
    errors = abs(Y_prediction - y_test)

    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
    
    #random_forest.score(X_test, Y_prediction)
    acc_xgb = round(xgb.score(X_test, Y_prediction) * 100, 2)
    print(round(acc_xgb,2,), "%")
    return xgb
