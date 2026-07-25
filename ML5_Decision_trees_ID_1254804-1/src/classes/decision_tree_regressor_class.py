import numpy as np
import pandas as pd
from classes.node_class import Node
from functions import find_best_split_randomized

class DecisionTreeRegressor():
    """
    A decision tree regressor for continuous target prediction.
    
    This regressor builds a binary decision tree using the standard deviation
    criterion to select the best splits. The prediction in each leaf is the
    mean value of the target samples that reached that leaf.
    
    Parameters
    ----------
    max_depth : int, optional (default=100)
        The maximum depth of the tree. Limits how deep the tree can grow.
    """
    
    def __init__(self, max_depth, max_features = None, random_state = None) -> None:
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.X = None
        self.y = None

    def _build_tree(self, X: pd.DataFrame, y: np.ndarray, data_indices: np.ndarray, depth: int) -> Node:
        """
        Recursively builds the decision tree by finding the best split at each node.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : np.ndarray
            Target values (continuous).
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
        

        if self.max_features is None:
            max_features = X.shape[1]
        else: max_features = self.max_features

        if (depth >= self.max_depth) or (np.std(y[data_indices]) == 0):
            node.is_leaf = True
            y_temp = y[data_indices]
            node.value = np.mean(y_temp)
            return node 
        else: 
            best_column, threshold = find_best_split_randomized(X, y, data_indices, max_features, criterion='std', random_state=self.random_state)

            if best_column is None:
                node.is_leaf = True
                y_temp = y[data_indices]
                node.value = np.mean(y_temp)
                return node
            mask = X[data_indices, best_column] <= threshold
            left_indices = data_indices[mask]
            right_indices = data_indices[~mask]
            node.left_child = self._build_tree(X, y , left_indices, depth + 1)
            node.right_child = self._build_tree(X,y, right_indices, depth + 1)
            node.feature_index = best_column
            node.threshold = threshold

            

        return node
    
    def _predict_single(self, row: np.ndarray) -> float:
        """
        Traverses the tree from root to leaf for a single sample and returns
        the predicted value stored in the leaf.
        
        Parameters
        ----------
        row : pd.Series
            A single sample (one row of the feature matrix).
            
        Returns
        -------
        float
            The predicted value (mean of target values in the leaf).
        """
        current_node = self.root
        while (current_node.is_leaf == False):
            if row[current_node.feature_index] <= current_node.threshold:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child
            
        return current_node.value

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'DecisionTreeRegressor':
        """
        Fits the decision tree regressor to the training data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training feature matrix.
        y : np.ndarray
            Target values (continuous).
            
        Returns
        -------
        DecisionTreeRegressor
            Returns self (the fitted regressor).
        """
        self.feature_names = X.columns
        self.X = X.values #конвертация для скорости
        self.y = y # уже numpy
        data_indices = np.arange(len(y)) # нужен массив индексов, а не просто число например 1000, а [0, 1, 2, 3, 4,5 и тд]
        self.root = self._build_tree(self.X, self.y, data_indices, depth=0) # depth - начальная текущая глубина( потом это увеличивается до max_depth)
        return self

    def predict_proba(self, X: pd.DataFrame) -> None:
        """
        Not applicable for regression. Returns None.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix of samples.
            
        Returns
        -------
        None
            This method is not implemented for regression tasks.
        """
        pass # для регрессии он не нужен

    def predict(self, X: pd.DataFrame) -> list:
        """
        Predicts target values for each sample in X.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix of samples to predict.
            
        Returns
        -------
        list
            A list of predicted values (floats), one per sample.
        """
        X_array = X.values     
        predictions = []
        for row in X_array:
            one_row_proba = self._predict_single(row)
            predictions.append(one_row_proba)

        
        return predictions