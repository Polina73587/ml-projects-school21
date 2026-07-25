import numpy as np
import pandas as pd
from functions import calculate_gini_impurity, find_best_split_randomized
from classes.node_class import Node


class ExtraTreesClassifier():
    """
    An Extra Trees classifier for binary classification tasks.
    
    This classifier builds a binary decision tree using the Extra Randomized Tree
    approach, where a random subset of features is selected at each split. This
    adds diversity to the model and helps prevent overfitting. It uses the Gini
    impurity criterion to select the best splits among the randomly chosen features.
    
    Parameters
    ----------
    max_depth : int
        The maximum depth of the tree. Limits how deep the tree can grow.
    max_feature : int
        Number of features to randomly select for split evaluation at each node.
        Must be between 1 and the total number of features in X.
    criterion : str
        The criterion to use for evaluating splits. Must be either:
        - 'gini': for classification tasks (uses Gini impurity)
        - 'std': for regression tasks (uses standard deviation)
    random_state : int or None, optional (default=None)
        Seed for random number generator to ensure reproducibility.
        If None, random selection will be different each time.
    
    Attributes
    ----------
    X : pd.DataFrame or None
        Training feature matrix (set during fit).
    y : np.ndarray or None
        Training target values (set during fit).
    root : Node
        Root node of the fitted decision tree.
    
    Examples
    --------
        >>> model = ExtraTreesClassifier(max_depth=7, max_feature=5, criterion='gini', random_state=42)
        >>> model.fit(Xtrain, ytrain)
        >>> predictions = model.predict(Xvalid)
        >>> probabilities = model.predict_proba(Xvalid)
    """
    
    def __init__(self, max_depth, max_feature, criterion, random_state) -> None:
        self.max_depth = max_depth
        self.X = None
        self.y = None
        self.max_feature = max_feature
        self.criterion = criterion
        self.random_state = random_state
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
            best_column, threshold = find_best_split_randomized(X, y, data_indices, self.max_feature, self.criterion, self.random_state)
            if best_column is None:
                node.is_leaf = True
                y_temp = y[data_indices]
                node.value = [ len(y_temp[y_temp == 0]) / len(y_temp), len(y_temp[y_temp == 1]) / len(y_temp) ]
                return node
            mask = X[data_indices, best_column] <= threshold
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
            feature_value = row.iloc[current_node.feature_index]
            if feature_value <= current_node.threshold:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child
            
        return current_node.value

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'ExtraTreesClassifier':
        """
        Fits the Extra Trees classifier to the training data.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training feature matrix.
        y : np.ndarray
            Target values.
            
        Returns
        -------
        ExtraTreesClassifier
            Returns self (the fitted classifier).
        """
       
        X_array = X.values if isinstance(X, pd.DataFrame) else X  # конвертируем в numpy array
        y_array = np.array(y)
        
        self.X = X_array
        self.y = y_array
        data_indices = np.arange(len(y_array))
        self.root = self._build_tree(self.X, self.y, data_indices, depth=0)
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
    