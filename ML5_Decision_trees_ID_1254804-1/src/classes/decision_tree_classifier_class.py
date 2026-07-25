import numpy as np
import pandas as pd
from functions import calculate_gini_impurity, find_best_split
from classes.node_class import Node

class DecisionTreeClassifier():
    """
    A decision tree classifier for binary classification tasks.
    
    This classifier builds a binary decision tree using the Gini impurity criterion
    to select the best splits. It supports configurable maximum depth to prevent
    overfitting.
    
    Parameters
    ----------
    max_depth : int, optional (default=100)
        The maximum depth of the tree. Limits how deep the tree can grow.
    """
    
    def __init__(self, max_depth: int = 100) -> None:
        self.max_depth = max_depth
        self.X = None
        self.y = None
        pass

    def _build_tree(self, X: pd.DataFrame, y: np.ndarray, data_indices: np.ndarray, depth: int) -> Node:
        """
        Recursively builds the decision tree by finding the best split at each node.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : np.ndarray
            Target values.
        data_indices : np.ndarray
            Indices of samples in the current node.
        depth : int
            Current depth of the node.
            
        Returns
        -------
        Node
            The constructed node (either a leaf or an internal node with children).
        """
        node = Node(y, data_indices)
        
        if (depth >= self.max_depth) or (calculate_gini_impurity(y[data_indices]) == 0):
            node.is_leaf = True
            y_temp = y[data_indices]
            node.value = [ len(y_temp[y_temp == 0]) / len(y_temp), len(y_temp[y_temp == 1]) / len(y_temp) ]
            return node 
        else: 
            best_column, threshold = find_best_split(X, y, data_indices)
            if best_column is None:
                node.is_leaf = True
                y_temp = y[data_indices]
                node.value = [ len(y_temp[y_temp == 0]) / len(y_temp), len(y_temp[y_temp == 1]) / len(y_temp) ]
                return node
            mask = X.loc[data_indices, best_column] <= threshold
            left_indices = data_indices[mask]
            right_indices = data_indices[~mask]
            node.left_child = self._build_tree(X, y , left_indices, depth + 1)
            node.right_child = self._build_tree(X,y, right_indices, depth + 1)
            node.feature_index = best_column
            node.threshold = threshold

        return node
    
    def _predict_single(self, row: pd.Series) -> list:
        """
        Traverses the tree from root to leaf for a single sample and returns
        the class probabilities stored in the leaf.
        
        Parameters
        ----------
        row : pd.Series
            A single sample (one row of the feature matrix).
            
        Returns
        -------
        list
            A list of class probabilities [p_class_0, p_class_1].
        """
        current_node = self.root
        while (current_node.is_leaf == False):
            if row[current_node.feature_index] <= current_node.threshold:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child
            
        return current_node.value

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'DecisionTreeClassifier':
        """
        Fits the decision tree classifier to the training data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training feature matrix.
        y : np.ndarray
            Target values.
            
        Returns
        -------
        DecisionTreeClassifier
            Returns self (the fitted classifier).
        """
        self.X = X
        self.y = y
        data_indices = np.arange(len(y)) # нужен массив индексов, а не просто число например 1000, а [0, 1, 2, 3, 4,5 и тд]
        self.root = self._build_tree(self.X, self.y, data_indices, depth=0) # depth - начальная текущая глубина( потом это увеличивается до max_depth)
        return self

    def predict_proba(self, X: pd.DataFrame) -> list:
        """
        Predicts class probabilities for each sample in X.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix of samples to predict.
            
        Returns
        -------
        list
            A list of class probability lists, one per sample.
            Each inner list has the form [p_class_0, p_class_1].
        """
        all_probas = []
        for idx, row in X.iterrows():
            one_row_proba = self._predict_single(row)
            all_probas.append(one_row_proba)
        return all_probas

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts class labels for each sample in X by selecting the class
        with the highest probability.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix of samples to predict.
            
        Returns
        -------
        np.ndarray
            Array of predicted class labels (0 or 1) for each sample.
        """
        probas = np.array(self.predict_proba(X))
        predictions = np.argmax(probas, axis=1) # axis=0  означает "по столбцам", axis=1 — "по строкам"
        return predictions