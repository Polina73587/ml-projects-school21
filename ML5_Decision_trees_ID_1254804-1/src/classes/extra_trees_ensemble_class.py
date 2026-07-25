import numpy as np
import pandas as pd
from classes.extra_trees_classifier_class import ExtraTreesClassifier


class ExtraTreesEnsemble:
    """Ансамбль из ExtraTreesClassifier (Extremely Randomized Trees)."""
    
    def __init__(self, n_estimators=100, max_depth=10, max_feature=5, random_state=21):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_feature = max_feature
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.trees = []
        
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = np.array(y)

        for i in range(self.n_estimators):
            indices = np.random.choice(len(X_array), size=len(X_array), replace=True)
            X_boot = X_array[indices]
            y_boot = y_array[indices]

            tree = ExtraTreesClassifier(
                max_depth=self.max_depth,
                max_feature=self.max_feature,
                criterion='gini',
                random_state=self.random_state + i 
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            
        return self

    def predict_proba(self, X):
        all_probas = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(all_probas, axis=0)

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)