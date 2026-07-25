import logging
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from sklearn.metrics import roc_auc_score


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimizes dataframe memory usage by downcasting data types.
    
    The function analyzes each column of the dataframe and downcasts it to the
    smallest possible data type that can correctly store all values in that column.
    This allows significant reduction of RAM usage without data loss.
    
    Logic:
    - For integer columns (int): selects the minimal type (int8, int16, int32, int64)
      based on the value range (min/max) in the column.
    - For float columns (float): selects between float32 and float64 depending
      on the value range.
    - For string columns (object/string): converts to 'category' type if the number
      of unique values is less than 50% of the total number of rows, which is
      efficient for columns with repeating values.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe to optimize memory for.
        
    Returns
    -------
    pd.DataFrame
        The same dataframe with optimized data types.
        
    Side Effects
    ------------
    - Logs the size before and after optimization, as well as the memory savings percentage.
    - Modifies the input dataframe in place (changes column data types).
    
    Example
    -------
        df = pd.read_csv('data.csv')
        df = reduce_memory_usage(df)
        # Size reduced from 72.65 MB to 5.89 MB (91.9% savings)
    """
        
    before = (df.memory_usage(deep=True).sum()) / 2**20
    logging.info('ЭТАП 1: ОПТИМИЗАЦИЯ ПАМЯТИ')
    logging.info(f"Размер до оптимизации: {before:.2f} MB")
    for col in df.columns:
        
        if pd.api.types.is_integer_dtype(df[col]):
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min >= -128 and col_max <= 127: col_type = 'int8'
            elif col_min >= -32768 and col_max <= 32767: col_type = 'int16'
            elif col_min >= -2147483648 and col_max <= 2147483647: col_type = 'int32'
            else: col_type = 'int64'
            df[col] = df[col].astype(dtype = col_type)
        elif pd.api.types.is_float_dtype(df[col]):
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min >= -2147483648 and col_max <= 2147483647: col_type = 'float32'
            else: col_type = 'float64'
            df[col] = df[col].astype(dtype = col_type)
        elif pd.api.types.is_string_dtype(df[col]):
            unique = df[col].nunique(dropna = True)
            total_rows = len(df)
            percentage = (unique/ total_rows) * 100
            if percentage < 50:
                df[col] = df[col].astype(dtype = 'category') 
    
    after = (df.memory_usage(deep=True).sum()) / 2**20
    
    logging.info(f"Размер после оптимизиции: {after:.2f} MB. Экономия: {(before-after)/before*100:.1f}%\n")
    return df



def calculate_gini_impurity(y_array: np.ndarray) -> float:
    """
    Calculates the Gini impurity of a binary target array.
    
    Gini impurity measures how often a randomly chosen element would be
    incorrectly labeled if it was randomly labeled according to the class
    distribution in the dataset. A Gini impurity of 0 indicates perfect
    purity (all elements belong to a single class), while a value of 0.5
    (for binary classification) indicates maximum impurity (classes are
    equally distributed).
    
    The formula used: Gini = 1 - (p_0^2 + p_1^2), where p_0 and p_1 are
    the proportions of class 0 and class 1 in the array respectively.
    
    Parameters
    ----------
    y_array : np.ndarray
        A binary target array containing values 0 and 1.
        
    Returns
    -------
    float
        The Gini impurity value, ranging from 0 (pure) to 0.5 (maximally impure).
    """
    p_0 = len(y_array[y_array == 0]) / len(y_array)
    p_1 = len(y_array[y_array == 1]) / len(y_array)
    gini = 1 - (p_0**2 + p_1**2)
    return gini



def find_best_split(X: pd.DataFrame, y: np.ndarray, data_indices: np.ndarray) -> Tuple[Optional[str], float]:
    """
    Finds the best possible split in the current node using all available features.
    
    This function evaluates all features and all possible thresholds to find the
    split that minimizes the weighted Gini impurity. It is used by the standard
    Decision Tree Classifier to determine the optimal split at each node.
    
    The function iterates through each feature, calculates all possible thresholds
    (midpoints between consecutive unique values), and evaluates the weighted Gini
    impurity for each split. The split with the lowest weighted Gini impurity is
    selected as the best split.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix containing all samples.
    y : np.ndarray
        Target values for all samples (binary: 0 or 1).
    data_indices : np.ndarray
        Indices of samples that belong to the current node.
        
    Returns
    -------
    Tuple[Optional[str], float]
        A tuple containing:
        - best_column (str or None): Name of the best feature for splitting.
          None if no valid split was found.
        - threshold (float): The threshold value for the best split.
        
    Raises
    ------
    ValueError
        If X or data_indices is empty.
        
    Examples
    --------
        >>> best_col, threshold = find_best_split(X, y, data_indices)
        >>> print(f"Best split: {best_col} <= {threshold}")
    """
    if len(X) == 0 or len(data_indices) == 0:
        raise ValueError("X and data_indices cannot be empty")
    
    best_gini = 1
    best_column = None
    threshold = 0
    
    for col in X.columns:
        unique = np.unique(X[col]) #возвращает отсорт. [ 1, 2, 3, 4 ,5]
        unique1 = unique[1:] # начинаем со второго индекса [ 2, 3, 4, 5]
        unique2 = unique[:-1] # все кроме последнего [ 1, 2, 3, 4]
        # таким образом получили два уникальных списка, чтобы удобно было делать векторизацию (это гениально я считаю )
        mean_unique_list  = (unique1 + unique2) / 2
        for j in mean_unique_list:
            mask = X.loc[data_indices, col] <= j
            left_indices = data_indices[mask]
            right_indices = data_indices[~mask]

            if len(left_indices) == 0 or len(right_indices) == 0:
                continue
            y_left = y[left_indices]
            y_right = y[right_indices]
            left_group_weight = len(left_indices) / len(data_indices)
            right_group_weight = len(right_indices) / len(data_indices)
            
            score_left = calculate_gini_impurity(y_left)
            score_right = calculate_gini_impurity(y_right)
         

            gini = left_group_weight * score_left + right_group_weight * score_right

            if gini < best_gini:
                best_gini = gini
                best_column = col
                threshold = j
    
    if best_column is None:
        logging.warning("No valid split found")

    return best_column, threshold

def find_best_split_randomized(X: pd.DataFrame, y: np.ndarray, data_indices: np.ndarray, max_feature: int, 
                               criterion: str, random_state: Optional[int] = None) -> Tuple[Optional[str], float]:
    """
    Finds the best split for a node using a random subset of features.
    
    This function implements the Extra Randomized Tree approach by randomly selecting
    a subset of features and finding the best split among them. This adds diversity
    to the ensemble and helps prevent overfitting.
    
    The function supports both classification (using Gini impurity) and regression
    (using standard deviation) criteria.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix containing all samples.
    y : np.ndarray
        Target values for all samples.
    data_indices : np.ndarray
        Indices of samples that belong to the current node.
    max_feature : int
        Number of features to randomly select for split evaluation.
        Must be between 1 and the total number of features in X.
    criterion : str
        The criterion to use for evaluating splits. Must be either:
        - 'gini': for classification tasks (uses Gini impurity)
        - 'std': for regression tasks (uses standard deviation)
    random_state : Optional[int], optional (default=None)
        Seed for random number generator to ensure reproducibility.
        If None, random selection will be different each time.
        
    Returns
    -------
    Tuple[Optional[str], float]
        A tuple containing:
        - best_column (str or None): Name of the best feature for splitting.
          None if no valid split was found.
        - threshold (float): The threshold value for the best split.
        
    Raises
    ------
    ValueError
        If max_feature is not between 1 and the number of features in X.
        If criterion is not 'gini' or 'std'.
        If X or data_indices is empty.
        
    Examples
    --------
        >>> best_col, threshold = find_best_split_randomized(
        ...     X, y, data_indices, max_feature=5, criterion='gini', random_state=42
        ... )
        >>> print(f"Best split: {best_col} <= {threshold}")
    """

    if len(X) == 0 or len(data_indices) == 0:
        raise ValueError("X and data_indices cannot be empty")
    
    X_array = X.values if isinstance(X, pd.DataFrame) else X
    
    n_features = X.shape[1] # количество колонок
    if max_feature < 1 or max_feature > n_features:
        raise ValueError(f"max_feature must be between 1 and {n_features}, got {max_feature}")
    
    if criterion not in ['gini', 'std']:
        raise ValueError(f"criterion must be 'gini' or 'std', got '{criterion}'")

    random_features = np.random.choice(n_features, max_feature, replace=False)
    best_score = float('inf')
    best_column = None
    threshold = 0
    
    logging.debug(f"Randomized split: evaluating {max_feature} random features: {random_features}")
    
    for col_index in random_features:
        unique = np.unique(X_array[data_indices, col_index])  # только строки текущего узла #возвращает отсорт. [ 1, 2, 3, 4 ,5]
        unique1 = unique[1:] # начинаем со второго индекса [ 2, 3, 4, 5]
        unique2 = unique[:-1] # все кроме последнего [ 1, 2, 3, 4]
        # таким образом получили два уникальных списка, чтобы удобно было делать векторизацию (это гениально я считаю )
        mean_unique_list  = (unique1 + unique2) / 2
        for j in mean_unique_list:
            mask = X_array[data_indices, col_index] <= j
            left_indices = data_indices[mask]
            right_indices = data_indices[~mask]

            if len(left_indices) == 0 or len(right_indices) == 0:
                continue
            y_left = y[left_indices]
            y_right = y[right_indices]
            left_group_weight = len(left_indices) / len(data_indices)
            right_group_weight = len(right_indices) / len(data_indices)
            if criterion == 'gini':
                score_left = calculate_gini_impurity(y_left)
                score_right = calculate_gini_impurity(y_right)
            elif criterion == 'std': 
                score_left = np.std(y_left)
                score_right = np.std(y_right)

            score = left_group_weight * score_left + right_group_weight * score_right

            if score < best_score:
                best_score = score
                best_column = col_index
                threshold = j
    
    if best_column is None:
        logging.debug("No valid split found in randomized features")
  
    return best_column, threshold



def sigmoid(x):
    """
    Вычисляет значение сигмоидальной (логистической) функции активации.

    Преобразует входные значения (логиты) в вероятности, отображая любое 
    действительное число в диапазон от 0 до 1. Функция полностью поддерживает 
    векторизованные операции, что позволяет эффективно обрабатывать как 
    скалярные значения, так и многомерные массивы данных.

    Args:
        x (float, int, np.ndarray, pd.Series или pd.DataFrame): Входные данные 
            (логиты), для которых необходимо вычислить значение сигмоиды. 
            Может быть как одиночным числом, так и структурой данных.

    Returns:
        float, np.ndarray, pd.Series или pd.DataFrame: Значения сигмоидальной 
            функции. Тип и форма возвращаемого объекта полностью соответствуют 
            типу и форме входных данных `x`. Все выходные значения строго 
            находятся в интервале (0, 1).
    """
    return 1 / (1 + np.exp(-x))

def show_results(model, X_test, y_test, min_threshold=None, verbose=True) -> float:
    """
    Оценивает качество работы классификационной модели на предоставленных данных.

    Функция вычисляет вероятности принадлежности к положительному классу, 
    рассчитывает метрику AUC-ROC и конвертирует её в Gini Score. 
    При включенном флаге `verbose` выводит результаты в консоль и системный лог, 
    а также выполняет проверку на достижение заданного минимального порога качества.

    Args:
        model: Обученная модель машинного обучения, поддерживающая метод `predict_proba` (например, из библиотек sklearn, catboost, lightgbm).
        X_test (array-like или pd.DataFrame): Матрица признаков тестовой или валидационной выборки.
        y_test (array-like или pd.Series): Вектор истинных меток классов (target) для соответствующей выборки.
        min_threshold (float, optional): Минимальное требуемое значение Gini Score для признания задачи выполненной. По умолчанию None.
        verbose (bool, optional): Флаг, включающий или отключающий вывод информации в консоль и логирование. По умолчанию True.

    Returns:
        float: Вычисленное значение метрики Gini Score (рассчитывается по формуле: 2 * AUC - 1).
    """
    probas = model.predict_proba(X_test)

    probas_array = np.array(probas)
    probas_positive = probas_array[:, 1]
    auc = roc_auc_score(y_test, probas_positive)
    gini = 2 * auc - 1

    if verbose:
        logging.info('Предсказание на validation:')
        print(f'Gini score: {gini:.4f}')
        if min_threshold is not None:
            logging.info(f'Gini score: {gini:.4f}')
            logging.info(f'Требуемый порог: {min_threshold}')
            if gini >= min_threshold: 
                logging.info(f'ЗАДАНИЕ ВЫПОЛНЕНО: Gini score ({gini:.4f}) >= {min_threshold}\n')
            else:
                logging.info(f'ЗАДАНИЕ НЕ ВЫПОЛНЕНО: Gini score ({gini:.4f}) < {min_threshold}\n')
        else: 
            logging.info(f'Gini score: {gini:.4f}\n')
   
    return float(gini)

def calculate_gini_score(model, model_name, X, y_true ) -> float:
    """
    Вычисляет Gini score для переданной модели.
    
    Параметры:
    model: обученный Pipeline
    X: датафрейм с признаками (например, valid_df)
    y_true: истинные значения целевой переменной (например, y_valid)
    
    Возвращает:
    float: значение Gini score
    """
    probs = model.predict_proba(X)[:, 1] # : - взять все строки, 1 - взять столбец с индексом 1 (второй столбец)
    auc = roc_auc_score(y_true, probs)
    gini = 2 * auc - 1
    logging.info(f'{model_name} Gini: {gini:.4f}')
    return gini

