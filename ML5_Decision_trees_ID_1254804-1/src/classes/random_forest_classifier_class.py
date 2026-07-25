import numpy as np
import pandas as pd
from functions import calculate_gini_impurity, find_best_split_randomized
from classes.node_class import Node


class RandomForestClassifier:
    """
    Random Forest Classifier implementation using Gini impurity criterion.
    
    This ensemble method builds multiple decision trees on bootstrap samples
    of the data and uses random feature subsets at each split to ensure
    diversity among trees. Predictions are made by averaging probabilities
    from all trees in the forest.
    
    Attributes
    ----------
    n_estimators : int
        Number of trees in the forest.
    max_depth : int
        Maximum depth of each tree.
    max_features : int
        Number of features to consider at each split.
    max_samples : float
        Fraction of samples to draw for each tree's bootstrap.
    random_state : int
        Seed for reproducibility.
    roots : list of Node
        List of root nodes for all trees in the forest.
    """

    def __init__(self, n_estimators: int = 10,max_depth: int = 5,max_features: int = 5,max_samples: float = 0.8,random_state =  None) -> None:
        """
        Initialize the Random Forest Classifier.
        
        Parameters
        ----------
        n_estimators : int, optional (default=10)
            Number of decision trees to build in the forest.
        max_depth : int, optional (default=5)
            Maximum depth allowed for each tree.
        max_features : int, optional (default=5)
            Number of random features to consider at each split.
        max_samples : float, optional (default=0.8)
            Fraction of the training set to use for each tree's bootstrap sample.
        random_state : int, optional (default=21)
            Seed for the random number generator to ensure reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state
        self.roots = []

    def _build_tree(self,X: np.ndarray,y: np.ndarray,data_indices: np.ndarray,depth: int) -> Node:
        """
        Recursively build a decision tree using randomized feature splits.
        
        At each node, a random subset of features is considered for splitting
        (via `find_best_split_randomized`). The recursion stops when the maximum
        depth is reached, the node becomes pure (Gini impurity equals 0), or
        no valid split can be found.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix containing all samples.
        y : np.ndarray
            Target values for all samples.
        data_indices : np.ndarray
            Indices of samples that belong to the current node (bootstrap sample).
        depth : int
            Current depth of the tree.
            
        Returns
        -------
        Node
            The root node of the constructed (sub)tree.
        """
        node = Node(y, data_indices)
        
        if (depth >= self.max_depth) or (calculate_gini_impurity(y[data_indices]) == 0):
            node.is_leaf = True
            y_temp = y[data_indices]
            node.value = [ len(y_temp[y_temp == 0]) / len(y_temp), len(y_temp[y_temp == 1]) / len(y_temp) ]
            return node 
        else: 
            best_column, threshold = find_best_split_randomized(X, y, data_indices, max_feature=self.max_features, criterion='gini', random_state=None)
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
    
    def _predict_single(self, row: np.ndarray, tree_root: Node) -> list:
        """
        Traverse a single tree to obtain class probabilities for one sample.
        
        Starting from the given root node, the method follows the decision
        path down to a leaf node based on feature thresholds.
        
        Parameters
        ----------
        row : np.ndarray
            A single sample (one row of the feature matrix).
        tree_root : Node
            The root node of the tree to traverse.
            
        Returns
        -------
        list
            A list of class probabilities [P(class_0), P(class_1)] stored
            in the leaf node.
        """
        current_node = tree_root
        while (current_node.is_leaf == False):
            if row[current_node.feature_index] <= current_node.threshold:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child
            
        return current_node.value

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'RandomForestClassifier':
        """
        Build the random forest by training `n_estimators` decision trees.
        
        Each tree is trained on a bootstrap sample of the data (rows sampled
        with replacement). At every split, a random subset of features is
        considered, which introduces diversity across trees and reduces
        overfitting.
        
        Parameters
        ----------
        X : pd.DataFrame
            Training feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target values of shape (n_samples,).
            
        Returns
        -------
        RandomForestClassifier
            Returns self (to allow method chaining).
        """
        X_array = X.values 
        if self.random_state is not None:
            np.random.seed(self.random_state)
        for i in range(self.n_estimators):
            bootstrap_indices = np.random.choice(X_array.shape[0], size = int(X_array.shape[0] * self.max_samples), replace=True) # индексы строк внутри X # True - можно использовать несколько раз.
            root = self._build_tree(X_array, y, data_indices = bootstrap_indices, depth=0)
            self.roots.append(root)

        return self

    def predict_proba(self, X: pd.DataFrame) -> list:
        """
        Predict class probabilities for each sample by averaging tree outputs.
        
        For every sample in X, the method traverses all trees in the forest,
        collects the class probabilities from each tree's leaf, and returns
        the element-wise mean across all trees.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix of shape (n_samples, n_features).
            
        Returns
        -------
        list
            A list of length n_samples, where each element is a numpy array
            of shape (2,) containing [P(class_0), P(class_1)].
        """
        X_array = X.values
        all_probas = []
        probas = []
        for row in X_array:
            tree_predictions = [self._predict_single(row, root) for root in self.roots]
            mean_proba = np.mean(tree_predictions, axis=0) # axis=0  означает "по столбцам", axis=1 — "по строкам"
            all_probas.append(mean_proba)
        return all_probas

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for each sample in X.
        
        The class with the highest averaged probability (from `predict_proba`)
        is selected as the prediction.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix of shape (n_samples, n_features).
            
        Returns
        -------
        np.ndarray
            Array of predicted class labels of shape (n_samples,).
        """
        probas = np.array(self.predict_proba(X))
        predictions = np.argmax(probas, axis=1) # axis=0  означает "по столбцам", axis=1 — "по строкам"
        return predictions