from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from cv_my import CVmy
import numpy as np

class ElasticNetMy():
    def __init__(self):
        pass
    def param_combinations(self, alpha, l1_ratio, X_train_scaled, y_train,  folds, all_results, best_score, best_params, best_model):
        fold_score = []
        for train_idx, test_idx in folds:
            X_train_fold = X_train_scaled[train_idx]
            X_val_fold = X_train_scaled[test_idx] # нет iloc потому что numpy-массив
            y_train_fold = y_train.iloc[train_idx]
            y_val_fold = y_train.iloc[test_idx] #iloc потому что pandas series 
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            r2 = r2_score(y_val_fold, y_pred)
            fold_score.append(r2)
        
        mean_score = np.mean(fold_score)
        all_results.append({
            'alpha': alpha,
            'l1_ratio': l1_ratio,
            'r2': mean_score
        })
        if mean_score > best_score:
            best_score = mean_score
            best_params = {'alpha': alpha, 'l1_ratio': l1_ratio}
            best_model = model

        return {
                'best_params': best_params,
                'best_score': best_score,
                'best_model': best_model,
                'all_results': all_results
                }
         
    def grid_search(self, alpha, l1_ratio, y_bins, X_train_scaled, y_train, random_state = None):
        best_score = -np.inf
        best_params = {}
        best_model = None
        all_results = []
        cv = CVmy()
        if random_state is not None:
            np.random.seed(random_state)
        folds = cv.stratified_K_fold(X_train_scaled, y_train, k = 5, stratify_field=y_bins, shuffle=True, random_state=random_state)
        for alph in alpha:
            for ratio in l1_ratio:
                res = self.param_combinations(alpha = alph, l1_ratio=ratio, X_train_scaled= X_train_scaled, y_train=y_train, folds=folds, all_results=all_results, best_score=best_score, 
                                    best_params=best_params, best_model=best_model)
            
               
                best_score = res['best_score']
                best_params = res['best_params']
                best_model = res['best_model']
                all_results = res['all_results']

        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_model': best_model,
            'all_results': all_results
        }

  
    def random_search(self, n_iter, X_train_scaled, y_train, y_bins, random_state = None):
        if random_state is not None:
            np.random.seed(random_state)
        cv = CVmy()
        best_score = -np.inf
        best_params = {}
        best_model = None
        all_results = []
        folds = cv.stratified_K_fold(X_train_scaled, y_train, k = 5, stratify_field=y_bins, shuffle=True, random_state=random_state)
        for i in range (n_iter):
            log_alpha = np.random.uniform(np.log(0.0001), np.log(100))
            alpha = np.exp(log_alpha)
            l1_ratio = np.random.uniform(0,1)
            res = self.param_combinations(alpha = alpha, l1_ratio=l1_ratio, X_train_scaled= X_train_scaled, y_train=y_train, folds=folds, all_results=all_results, best_score=best_score, 
                                    best_params=best_params, best_model=best_model)
            
               
            best_score = res['best_score']
            best_params = res['best_params']
            best_model = res['best_model']
            all_results = res['all_results']

        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_model': best_model,
            'all_results': all_results
        }