# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 22:23:47 2018

@author: 51648
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy import interp
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from pprint import pprint
from time import time
import logging
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import ensemble
from sklearn.svm import SVC

def Category(x):
    x = str(x)
    x = (x.split(','))[0]
    if x in ('Asim' , 'KV' , 'Mojtaba' , 'Matthew', 'Andrea'):
        return 1
    if x in ('Yiwen'):
        return 2
    if x in ( 'Olya', 'Olga', 'Sofia'):
        return 3
    if x in ('Maria', 'Casper'):
        return 4
    if x in ('Stefano','Lavinia','Sandra'):
        return 5
    if x in ('Jean' ,'Wen Li','Sireesha','Anja'):
        return 6
    if x in ('Nathan', 'Shenshen', 'Pim', 'Hanyu', 'Kajeng', 'Valeria', 'Jan Willem'):
        return 7
    if x in ('Bruno', 'Davi'):
        return 8
    

def Time(x):
    try: 
        x = str(x)
        x = (x.split(':'))[0]
        x = int(x)
        if (x > 9 and x < 18):
            return "CET"
        if (x > 18 and x < 24):
            return "CDT"
        if (x > 0 and x < 3):
            return "CDT"
        if (x > 3 and x < 9):
            return "SGT"    
    except:
        x = str(x)
        

def DomName(x):
    readCSV = csv.reader(open('country.csv', 'r'))
    DomDict = {rows[0]:rows[1] for rows in readCSV}
    x = str(x)
    x = x.upper()
    if x in DomDict:
        return DomDict[x]
    else:
        DomDict[x]="NULL"
        return DomDict[x]

def CleanStr(x):
    x = x.split(" ")
    d = []
    for mem in x:
        if "@" in mem:
            z =mem.split("@")
            z=z[0]
            d.append(z)
        else:
            d.append(mem)
    return d
#########################Pipeline for feature detection and evluation###########
def PipeFeauture(Xtrain, Ytrain):
    pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
    ])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        #'tfidf__use_idf': (True, False),
        #'tfidf__norm': ('l1', 'l2'),
        'clf__alpha': (0.00001, 0.000001),
        'clf__penalty': ('l2', 'elasticnet'),
        #'clf__n_iter': (10, 50, 80),
    }
    
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(Xtrain, Ytrain)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

#*********************** Optimize Hyper Parameters *******************************
def HyperParamRFC(x,y):
    from sklearn.cross_validation import train_test_split
    y = label_binarize(y, classes=[1, 2, 3, 4, 5, 6, 7, 8])
    n_classes = y.shape[1]
    ## Spltting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)
    
    from sklearn.ensemble import RandomForestClassifier
    #my_pipeline = make_pipeline(RandomForestClassifier(n_jobs=2, random_state=80, n_estimators = 20))
    
    #### Using RANDOM Search CV to find the best hyper parameters
    from sklearn.model_selection import RandomizedSearchCV
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    
    rf = RandomForestClassifier()
    rf_random = GridSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 25, cv = 3, verbose=2, random_state=42, n_jobs = 1)
    
    
    #classifier = GaussianNB()
    #my_pipeline.fit(X_train, y_train)
    rf_random.fit(X_train, y_train)
    print("best parameters are",rf_random.best_params_)
    
    #### Cross-validation to check the test set error rate
    scores = cross_val_score(rf_random, X_train, y_train, scoring='neg_mean_absolute_error')
    print(scores)
    
    
    ## Predicting the test set results
    y_pred = rf_random.predict(X_test)
    return y_pred
##############################################################################
#def HyperParamSGD():
    
    
###################### ROC and Precision-Recall plot #########################

def AccuPlot(y, y_test, y_pred):
        # this part is more of plotting ROC, Precision, and Recall
    y = label_binarize(y, classes=[1, 2, 3, 4, 5, 6, 7, 8])
    n_classes = y.shape[1]
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_pred[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_pred[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_pred.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_pred, average="micro")
#    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))
#    plt.figure()
#    plt.subplot(221)
#    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2, where='post')
#    plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2, color='b')
#    plt.xlabel('Recall')
#    plt.ylabel('Precision')
#    plt.ylim([0.0, 1.05])
#    plt.xlim([0.0, 1.0])
#    plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))
    
    
    colors = cycle(['red','green','yellow','navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
    
    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))
    
    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))
    
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    #plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.legend(lines, labels, bbox_to_anchor=(1, 0.5), loc='center left', prop = dict(size=14))
 
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #Compute ROC
    from sklearn.metrics import roc_curve, auc
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#    plt.subplot(222)
#    lw = 2
#    plt.plot(fpr[2], tpr[2], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
#    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#    plt.xlim([0.0, 1.0])
#    plt.ylim([0.0, 1.05])
#    plt.xlabel('False Positive Rate')
#    plt.ylabel('True Positive Rate')
#    plt.title('Receiver operating characteristic example')
#    plt.legend(loc="lower right")
#    plt.show()
    
    
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure()
    plt.subplot(224)
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    lw = 2
    colors = cycle(['red','blue','yellow','navy','green','aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    #plt.legend(loc="lower right")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

##################### main function ##########################################
def main():
    df = pd.read_csv('C:\\Users\\51648\\Desktop\\Python\\OutlookVS\\CollectData\\TestEmails.csv')
    #print(df.columns)
    df = df.drop(['Flag','Length','Message Class', 'Body'], axis = 1)
    #print(df.columns)
    #df = df.drop(df.columns[0], axis = 1)
    #print(df.columns)
    #df.head()
    
    # here is to create the dataset from raw data
    from sklearn.preprocessing import Imputer
    imputer = Imputer()
    
    
    df['DomCon'] = df.domain.apply(DomName)
    df['SendEmailCon'] = df.SenderEmailAddress.apply(CleanStr)
    df['TimeCat'] = df.Time.apply(Time)
    df['CatgCat'] = df.Categories.apply(Category)
    df['combined'] = df.apply(lambda x: [[x.ToEmail, x.SenderName, x.CC, x.DomCon, x.Company, x.TimeCat]], axis=1)
    NewDf = df[['combined','CatgCat']].copy()
    NewDf.head()
    # here you have to remove nan from your dataset.
    
    NewDf = NewDf.dropna()
    #NewDf = imputer.fit_transform(NewDf)
    
    
    #test =str(NewDf['combined'][1]).strip('[]')
    #print(test)
    # Cleaning the data
    import re
    import nltk
    nltk.download('stopwords')
    corpus = []
    NewDf2 = NewDf[['combined']].copy()
    size = len(NewDf)
    
    for row in NewDf2.T.iteritems():
        #NewDf['combined'][i] = str(NewDf['combined'][i]).strip('[]')
        #NewDf2['combined'][i] = ' '.join(map(str,NewDf2['combined'][i]))
        
        row = ' '.join(map(str,row))
        review = re.sub('[^A-Za-z]',' ', row)
        review = re.sub (',', ' ', review)
        review = re.sub (';',' ', review)
        review = review.lower()
        review = review.split()
        #print(review)
        ps = PorterStemmer()
        #operators = set(('es,'))
        stop_words = set(stopwords.words('english'))
        stop_words.update(['ul', 'ts', 'fa' , 'null' , '!' , ':', ';', 'co', 'nan', '[', ']', '{', '}', ',', 'combined', 'name', 
                           'dtype', 'object', 'fas','p','b','e', 'g', 'a', 'b', 'c', 'd', 'f', 'h', 'i', 'j',
                      'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
                      'x', 'y', 'z'])
        
    #    stop_words2 = ['ul', 'ts','fa', 'null', '!', ':', ';', 'co', 'nan', '[', ']',
    #                  '{', '}', ',','e', 'g', 'a', 'b', 'c', 'd', 'f', 'h', 'i', 'j',
    #                  'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w',
    #                  'x', 'y', 'z', 'NULL']
        #review = [ps.stem(word) for word in review if not word in stopwords2]
        review = [word for word in review if not word in stop_words]
        review = ' '.join(review)
        corpus.append(review)
        
    # Creating the bag of words model-1
    #from sklearn.feature_extraction.text import CountVectorizer
    #cv = CountVectorizer(max_features = 200)
    #X = cv.fit_transform(corpus).toarray()
    #y = NewDf.iloc[:,1].values
    
    # different methods tfidf model - 2
    cv = TfidfVectorizer(min_df = 1, ngram_range = (1,2))
    X = cv.fit_transform(corpus).toarray()
    y = NewDf.iloc[:,1].values
    #y = label_binarize(y, classes=[1, 2, 3, 4, 5, 6, 7, 8])
    #n_classes = y.shape[1]
    
    #
    ## Spltting the dataset into the Training set and Test set
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
    #
    #
    ## Fitting Classifier to the training set
    ## Creating the classifier
    # Classifier 1
    #from sklearn.naive_bayes import GaussianNB
    #my_pipeline = make_pipeline(GaussianNB())
    
    #Classifier 2
    #from sklearn.naive_bayes import MultinomialNB
    #my_pipeline = make_pipeline(MultinomialNB(alpha = 0.5))
    
    # Classifier 3
    #from sklearn.ensemble import RandomForestClassifier
    #my_pipeline = make_pipeline(RandomForestClassifier(n_jobs=2, random_state=40, n_estimators = 2000))
    
    
    #Classifier 4 almost 75%
    #from sklearn import linear_model
    #my_pipeline = make_pipeline(linear_model.SGDClassifier(loss="log", penalty="l2", max_iter=3000))
    
    
    #Classifier 5
    #params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
    #      'learning_rate': 0.01, 'loss': 'ls'}
    #my_pipeline =make_pipeline(ensemble.GradientBoostingRegressor(**params))
    
    #Classifier 6
    #from sklearn import linear_model
    #my_pipeline = make_pipeline(SVC(kernel='linear', class_weight={1:10}, C=1.0, random_state=0))
    
    
    #Classifier 5 80%
    from sklearn import linear_model
    my_pipeline = make_pipeline(linear_model.LogisticRegression(C=1e6, class_weight= 'auto', solver='saga', max_iter = 1000))
    
    
    
    # check the data being balanced:
    scores1 = cross_val_score(my_pipeline, X, y, cv = 5)
    scores2 = cross_val_score(my_pipeline, X, y, cv = 5, scoring='f1_macro')
    print("This is the scores1 from cross validation = {}".format(scores1))
    print("This is the scores2 from cross validation = {}".format(scores2))
    

    
    #classifier = GaussianNB()
    my_pipeline.fit(X_train, y_train)
    #### Cross-validation to check the test set error rate
    k_fold = KFold(n_splits=3)
    scores = cross_val_score(my_pipeline, X_train, y_train, cv = k_fold ,scoring='neg_mean_absolute_error')
    print("This is the scores from cross validation = {}".format(scores))
    
    ## Predicting the test set results
    y_pred = my_pipeline.predict(X_test)   
    print("The accuracy score of {0}".format(accuracy_score(y_test, y_pred)))
  
    
    ##Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("This is the trace of cm matrix = {0}".format(np.trace(cm)))
    
    
    
    # check the accuracy, precision and recall.
    
    #+++++++++++++++++++ Multi Class problem
#    average_precision = average_precision_score(y_test, y_pred)
#    print('Average precision-recall score: {0:0.2f}'.format(
#      average_precision))
    
    # easy report on recall and precision
    print("This is the result of precision and recall:")
    print(classification_report(y_test, y_pred))
    #+++++++++++++++++++++++++++++++++++++++++++++++++
    
    
    
    
if __name__ == "__main__":
    main()
    
    
    
    
    


    
