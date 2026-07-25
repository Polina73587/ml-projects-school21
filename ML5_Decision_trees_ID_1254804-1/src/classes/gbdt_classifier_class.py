from classes.decision_tree_regressor_class import DecisionTreeRegressor
from functions import sigmoid
import numpy as np


class GBDTClassifier:
    def __init__(self, max_depth, number_of_trees, max_features, learning_rate, random_state = None):
        self.max_depth = max_depth
        self.number_of_trees = number_of_trees
        self.max_features = max_features
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        p = np.mean(y)
        F_0 = np.log( p / ( 1-p))
        self.F_0 = F_0
        F = np.full(len(y), F_0) # numpy array
        self.trees_list = []
        for i in range(self.number_of_trees):
            p = sigmoid(F)
            residuals = y - p
            tree = DecisionTreeRegressor(max_depth=self.max_depth, max_features=self.max_features, random_state=self.random_state)
            tree.fit(X,residuals)
            F = F + self.learning_rate * np.array(tree.predict(X))
            self.trees_list.append(tree)
        return self
    
    def predict_proba(self, X):
        F_0 = self.F_0
        F = np.full(len(X), F_0)  
        for tree in self.trees_list:
            F = F + self.learning_rate * np.array(tree.predict(X)) #len(X) элементов!
        probas = sigmoid(F) #массив вероятностей для всех объектов, len(F) элементов
        return np.column_stack([ 1 - probas, probas])#склеивает несколько массивов в столбцы.


    def predict(self, X):
        probas = self.predict_proba(X)
        predictions = np.argmax(probas, axis = 1) # axis=0  означает "по столбцам", axis=1 — "по строкам"
        return predictions

