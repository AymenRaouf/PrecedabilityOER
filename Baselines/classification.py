from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from itertools import product
from sklearn.svm import SVC
import pandas as pd

def classify(X,Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)
    
    print("Decision Tree :")
    decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
    y_pred = decision_tree.fit(X_train, y_train).predict(X_test)
    print("Number of mislabeled points out of a total %d points : %d"
        % (X_test.shape[0], (y_test != y_pred).sum()))
    print((1 - (y_test != y_pred).sum().item() / (X_test.shape[0])))
    print("====================================================") 

    print("SVM :")
    svm = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    y_pred = svm.fit(X_train, y_train).predict(X_test)
    print("Number of mislabeled points out of a total %d points : %d"
        % (X_test.shape[0], (y_test != y_pred).sum()))
    print((1 - (y_test != y_pred).sum().item() / (X_test.shape[0])))
    print("====================================================") 

    print("Gaussian Naive Bayes :")
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print("Number of mislabeled points out of a total %d points : %d"
        % (X_test.shape[0], (y_test != y_pred).sum()))
    print((1 - (y_test != y_pred).sum().item() / (X_test.shape[0])))
    print("====================================================") 

    print("KNN :")
    neigh = KNeighborsClassifier()
    y_pred = neigh.fit(X_train, y_train).predict(X_test)
    print("Number of mislabeled points out of a total %d points : %d"
        % (X_test.shape[0], (y_test != y_pred).sum()))
    print((1 - (y_test != y_pred).sum().item() / (X_test.shape[0])))
    print("====================================================") 



def classify_cv(X,Y):

    results = []
    
    result = {}
    print("Logistic Regression")
    param_logistic_regression = {
        'random_state' : [0],   
        'max_iter': [300], 
        }
    logistic_regression = GridSearchCV(LogisticRegression(), param_logistic_regression, cv = 5, scoring = 'accuracy')
    logistic_regression.fit(X,Y)
    scores = cross_val_score(logistic_regression.best_estimator_, X, Y, cv = 5)
    print("%0.2f mean accuracy\n %0.2f max accuracy\n %0.2f min accuracy\n %0.2f standard deviation" % 
        (scores.mean(), scores.max(), scores.min(), scores.std()))
    print(logistic_regression.best_params_)
    result['Accuracy Max'] = round(scores.max(), 2)
    result['Accuracy Avg'] = round(scores.mean(), 2)
    result['Accuracy Min'] = round(scores.min(), 2)
    result['Model'] = "Logistic Regression"
    results.append(result)
    print("====================================================")

    result = {}
    print("Decision Tree :")
    param_decision_tree = {
        'random_state' : [0],
        'max_depth': [1, 2, 5, 10], 
        'criterion': ['gini', 'entropy', 'log_loss'], 
        'splitter' : ['best', 'random']
        }
    decision_tree = GridSearchCV(DecisionTreeClassifier(), param_decision_tree, cv = 5, scoring = 'accuracy')
    decision_tree.fit(X,Y)
    scores = cross_val_score(decision_tree.best_estimator_, X, Y, cv = 5)
    print("%0.2f mean accuracy\n %0.2f max accuracy\n %0.2f min accuracy\n %0.2f standard deviation" % 
        (scores.mean(), scores.max(), scores.min(), scores.std()))
    print(decision_tree.best_params_)
    result['Accuracy Max'] = round(scores.max(), 2)
    result['Accuracy Avg'] = round(scores.mean(), 2)
    result['Accuracy Min'] = round(scores.min(), 2)
    result['Model'] = "Decision Tree"
    results.append(result)
    print("====================================================")

    result = {}
    print("Random Forest :")
    param_random_forest = {
        'random_state' : [0],
        'max_depth': [1, 2, 5, 10], 
        'criterion': ['gini', 'entropy', 'log_loss'], 
        'n_estimators' : [50, 100, 150, 200]
        }
    random_forest = GridSearchCV(RandomForestClassifier(), param_random_forest, cv = 5, scoring = 'accuracy')
    random_forest.fit(X,Y)
    scores = cross_val_score(random_forest.best_estimator_, X, Y, cv = 5)
    print("%0.2f mean accuracy\n %0.2f max accuracy\n %0.2f min accuracy\n %0.2f standard deviation" % 
        (scores.mean(), scores.max(), scores.min(), scores.std()))
    print(random_forest.best_params_)
    result['Accuracy Max'] = round(scores.max(), 2)
    result['Accuracy Avg'] = round(scores.mean(), 2)
    result['Accuracy Min'] = round(scores.min(), 2)
    result['Model'] = "Random Forest"
    results.append(result)
    print("====================================================")
    
    result = {}
    print("SVM :")
    param_svm = {
        'C': [1, 10, 100, 1000], 
        'kernel': ['linear']
    }
    svm = GridSearchCV(SVC(), param_svm, cv = 5, scoring='accuracy')
    svm.fit(X,Y)
    scores = cross_val_score(svm.best_estimator_, X, Y, cv = 5)
    print("%0.2f mean accuracy\n %0.2f max accuracy\n %0.2f min accuracy\n %0.2f standard deviation" % 
        (scores.mean(), scores.max(), scores.min(), scores.std()))
    print(svm.best_params_)
    result['Accuracy Max'] = round(scores.max(), 2)
    result['Accuracy Avg'] = round(scores.mean(), 2)
    result['Accuracy Min'] = round(scores.min(), 2)
    result['Model'] = "SVM"
    results.append(result)
    print("====================================================")

    result = {}
    print("RBF :")
    param_svm = {
        'C': [1, 10, 100, 1000], 
        'gamma': [0.1, 0.01, 0.001, 0.0001], 
        'kernel': ['rbf']
    }
    svm = GridSearchCV(SVC(), param_svm, cv = 5, scoring='accuracy')
    svm.fit(X,Y)
    scores = cross_val_score(svm.best_estimator_, X, Y, cv = 5)
    print("%0.2f mean accuracy\n %0.2f max accuracy\n %0.2f min accuracy\n %0.2f standard deviation" % 
        (scores.mean(), scores.max(), scores.min(), scores.std()))
    print(svm.best_params_)
    result['Accuracy Max'] = round(scores.max(), 2)
    result['Accuracy Avg'] = round(scores.mean(), 2)
    result['Accuracy Min'] = round(scores.min(), 2)
    result['Model'] = "RBF"
    results.append(result)
    print("====================================================")


    result = {}
    print("Gaussian Naive Bayes :")
    gnb = GaussianNB()
    gnb.fit(X,Y)
    scores = cross_val_score(gnb, X, Y, cv = 5)
    print("%0.2f mean accuracy\n %0.2f max accuracy\n %0.2f min accuracy\n %0.2f standard deviation" % 
        (scores.mean(), scores.max(), scores.min(), scores.std()))
    #print(gnb.best_params_)
    result['Accuracy Max'] = round(scores.max(), 2)
    result['Accuracy Avg'] = round(scores.mean(), 2)
    result['Accuracy Min'] = round(scores.min(), 2)
    result['Model'] = "GNB"
    results.append(result)
    print("====================================================")

    result = {}
    print("KNN :")
    param_knn = {
        'n_neighbors': [5, 7, 10, 20], 
        'weights': ['uniform', 'distance']
    }
    knn = GridSearchCV(KNeighborsClassifier(), param_knn, cv = 5, scoring='accuracy')
    knn.fit(X,Y)
    scores = cross_val_score(knn.best_estimator_, X, Y, cv = 5)
    print("%0.2f mean accuracy\n %0.2f max accuracy\n %0.2f min accuracy\n %0.2f standard deviation" % 
        (scores.mean(), scores.max(), scores.min(), scores.std()))
    print(knn.best_params_)
    result['Accuracy Max'] = round(scores.max(), 2)
    result['Accuracy Avg'] = round(scores.mean(), 2)
    result['Accuracy Min'] = round(scores.min(), 2)
    result['Model'] = "KNN"
    results.append(result)
    print("====================================================")
    return results


def classify_cv_predefined(X_train, Y_train, X_test, Y_test, cv):

    df_results = pd.DataFrame()
    for n in range(cv):
        
        X_train_batch = X_train[n]
        Y_train_batch = Y_train[n]
        X_test_batch = X_test[n]
        Y_test_batch = Y_test[n]
        
        print("Linear Regression :")
        results = {}
        logistic_regression = LogisticRegression(max_iter=300).fit(X_train_batch, Y_train_batch)
        Y_pred = logistic_regression.predict(X_test_batch)
        accuracy = accuracy_score(Y_test_batch, Y_pred)
        results['Model'] = "Linear Regression"
        results['Accuracy'] = accuracy
        results['Params'] = "None"
        df_results = pd.concat([df_results, pd.DataFrame([results])], ignore_index = True)

        print("SVM :")
        param_svm = {
            'C': [1, 10, 100, 1000], 
            'gamma': [0.1, 0.01, 0.001, 0.0001],
            'kernel': ['linear']
        }
        param_svm = list(product(*param_svm.values()))
        for params in param_svm:
            results = {}
            C, gamma, kernel = params
            svm = make_pipeline(StandardScaler(), SVC(gamma = gamma, C = C, kernel = kernel)).fit(X_train_batch, Y_train_batch)
            Y_pred = svm.predict(X_test_batch)
            accuracy = accuracy_score(Y_test_batch, Y_pred)
            results['Model'] = "SVM"
            results['Accuracy'] = accuracy
            results['Params'] = str(params)
            df_results = pd.concat([df_results, pd.DataFrame([results])], ignore_index = True)

        print("RBF :")
        param_rbf = {
            'C': [1, 10, 100, 1000], 
            'gamma': [0.1, 0.01, 0.001, 0.0001], 
            'kernel': ['rbf']
        }
        param_rbf = list(product(*param_rbf.values()))
        for params in param_rbf:
            results = {}
            C, gamma, kernel = params
            rbf = make_pipeline(StandardScaler(), SVC(gamma = gamma, C = C, kernel = kernel)).fit(X_train_batch, Y_train_batch)
            Y_pred = rbf.predict(X_test_batch)
            accuracy = accuracy_score(Y_test_batch, Y_pred)
            results['Model'] = "RBF"
            results['Accuracy'] = accuracy
            results['Params'] = str(params)
            df_results = pd.concat([df_results, pd.DataFrame([results])], ignore_index = True)

        print("Random Forest :")
        param_random_forest = {
            'random_state' : [0],
            'max_depth': [1, 2, 5, 10], 
            'n_estimators' : [50, 100, 150, 200],
            'criterion': ['gini', 'entropy', 'log_loss']            
        }
        param_random_forest = list(product(*param_random_forest.values()))
        for params in param_random_forest:
            results = {}
            random_state, max_depth, n_estimators, criterion = params
            rf = RandomForestClassifier(n_estimators = n_estimators, criterion = criterion, random_state = random_state, max_depth = max_depth ).fit(X_train_batch, Y_train_batch)
            Y_pred = rf.predict(X_test_batch)
            accuracy = accuracy_score(Y_test_batch, Y_pred)
            results['Model'] = "RF"
            results['Accuracy'] = accuracy
            results['Params'] = str(params)
            df_results = pd.concat([df_results, pd.DataFrame([results])], ignore_index = True)

    return df_results