import numpy as np
import pandas as pd

class CVmy():
    def __init__(self):
        pass
    def K_fold(self, X, y, k, shuffle = True, random_state = None):
        if len(X) != len(y):
            raise ValueError('X and y have different lengths')
        if k <= 1:
            raise ValueError('k value must be more that 1')
        if random_state is not None:
            np.random.seed(random_state)

        indices = list(range(len(X)))
        if shuffle:
            np.random.shuffle(indices)
        fold_size = len(X) // k 
        rem = len(X) % k
        current_pos = 0
        folds = []
        for i in range(k):
            size = fold_size + 1 if i < rem else fold_size
            fold = indices[current_pos: current_pos + size]
            folds.append(fold)
            current_pos += size

        result = []    
        for i in range (k):
            test_indices = folds[i]
            train_indices = []
            for j in range(k):
                if j != i:
                    train_indices.extend(folds[j])
            result.append((train_indices, test_indices))        
        return result
    
#--------------------------------------------------------------------------------

    def grouped_K_fold(self, X,y, k, group_field, shuffle = True, random_state = None):
        if len(X) != len(y):
            raise ValueError('X and y have different lengths')
        if group_field not in X.columns:
            raise ValueError('group field not in X features')
        if k <= 1:
            raise ValueError('k value must be more that 1')
        if random_state is not None:
            np.random.seed(random_state)

        unique_groups = set( X[group_field]) # перемешиваю доп, set не предназначен для случайных разбиений

        group_list = list(unique_groups)
        if shuffle:
            np.random.shuffle(group_list)

        fold_size = len(group_list) // k
        rem = len(group_list) % k 
        current_pos = 0
        folds = []
        for i in range(k):
            size = fold_size + 1 if i < rem else fold_size
            fold = group_list[current_pos : current_pos + size]
            folds.append(fold)
            current_pos += size

        result = []
        for i in range (k):
            test_groups = folds[i]
            mask = X[group_field].isin(test_groups)
            test_indices = np.where(mask)[0].tolist()
            train_groups = []
            for j in range (k):
                if j != i:
                    train_groups.extend(folds[j])
            mask_train = X[group_field].isin(train_groups)
            train_indices = np.where(mask_train)[0].tolist()
            result.append((train_indices, test_indices))
        return result
    
    #--------------------------------------------------------------------------------

    def stratified_K_fold(self, X, y, k, stratify_field, shuffle = True, random_state = None):
        if len(X) != len(y):
            raise ValueError('X and y have different lengths')
        if k <= 1 :
            raise ValueError('k value must be more that 1')
        
        if random_state is not None:
            np.random.seed(random_state)

        if isinstance(stratify_field, str):
            value = X[stratify_field]
        elif isinstance(stratify_field, (pd.Series, np.ndarray, list)):
            value = stratify_field

        value = np.array(value)
        unique_classes = np.unique(value)
    
        parts = []
        for uniq_class in unique_classes:
            part =  np.where( value == uniq_class)
        
            parts.append(part[0]) # только индексы ( имеем кортеж из массива индексов)

        fold_by_class = []
        for part in parts:
            if shuffle:
                np.random.shuffle(part)
            n = len(part)
            fold_size = n // k
            rem = n % k
            current_pos = 0 
            folds_for_this_class = []
            for i in range (k):
                size = fold_size + 1 if i < rem else fold_size
                fold = part[current_pos : current_pos + size]
                folds_for_this_class.append(fold)
                current_pos += size
            fold_by_class.append(folds_for_this_class)
        
        result = []
        for i in range(k):
            test_indices = []
            train_indices = []
            for class_folds in fold_by_class:
                test_indices.extend(class_folds[i])
                for j in range(k):
                    if j != i:
                        train_indices.extend(class_folds[j])
            result.append((train_indices, test_indices))
        return result
        
    #--------------------------------------------------------------------------------

    def time_series_split(self, X, y, k, date_field,random_state = None):
        if len(X) != len(y): 
            raise ValueError('X and y have different lengths')
        if k <= 1:
            raise ValueError('k value must be more that 1')
        if random_state is not None:
            np.random.seed(random_state)
        if date_field not in X.columns:
            raise ValueError('date field not in X features')
        
        sorted_indeces = X[date_field].sort_values().index
        X = X.loc[sorted_indeces]
        y = y.loc[sorted_indeces]

        n = len(X)
        test_size = n // k

        result = []
        for i in range(k):
            test_start = i * test_size
            test_end = min(test_start + test_size, n) # чтобы не выйти за границы
            train_indices = list(range(test_start))
            test_indices = list(range(test_start, test_end))
            result.append((train_indices, test_indices))
        return result