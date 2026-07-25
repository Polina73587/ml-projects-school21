import numpy as np
from functions import calculate_gini_impurity

class Node():
    """
    A node in the decision tree.
    
    Each node represents a decision point in the tree. Internal nodes contain
    split information (feature and threshold) and pointers to child nodes.
    Leaf nodes contain the predicted values (class probabilities for classification
    or mean values for regression).
    
    Attributes
    ----------
    data_indices : np.ndarray
        Indices of samples that belong to this node.
    feature_index : str or None
        Name of the feature used for splitting (None for leaf nodes).
    threshold : float or None
        Threshold value for the split (None for leaf nodes).
    left_child : Node or None
        Left child node (samples where feature <= threshold).
    right_child : Node or None
        Right child node (samples where feature > threshold).
    value : list or None
        For leaf nodes: class probabilities [p_class_0, p_class_1] for classification,
        or mean value for regression. None for internal nodes.
    is_leaf : bool
        True if this is a leaf node (no further splitting), False otherwise.
    y : np.ndarray
        Complete target array (used for calculations).
    """
    
    def __init__(self, y: np.ndarray, data_indices: np.ndarray) -> None:
        self.data_indices = data_indices
        self.feature_index = None
        self.threshold = None
        self.left_child = None
        self.right_child = None
        self.value = None
        self.is_leaf = False
        self.y = y


    def gini_impurity(self) -> float:
        """
        Computes the Gini impurity of the current node based on its target values.
        
        The Gini impurity measures how mixed the classes are in the node.
        A value of 0 means all samples belong to a single class (perfect purity),
        while a value of 0.5 (for binary classification) means classes are
        equally distributed.
        
        Returns
        -------
        float
            The Gini impurity value for the samples in this node.
        """
        y_temp = self.y[self.data_indices]
        return calculate_gini_impurity(y_temp)
  