import logging
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np

def reduce_memory_usage(df) -> pd.DataFrame:
    """
    Оптимизирует использование памяти датафрейма за счёт понижения типов данных.
    
    Функция анализирует каждую колонку датафрейма и приводит её к минимально возможному
    типу данных, который может корректно хранить все значения в этой колонке.
    Это позволяет значительно сократить объём занимаемой оперативной памяти без потери данных.
    
    Логика работы:
    - Для целочисленных колонок (int): выбирает минимальный тип (int8, int16, int32, int64)
      на основе диапазона значений (min/max) в колонке.
    - Для вещественных колонок (float): выбирает между float32 и float64 в зависимости
      от диапазона значений.
    - Для строковых колонок (object/string): конвертирует в тип 'category', если количество
      уникальных значений составляет менее 50% от общего числа строк, что эффективно для
      колонок с повторяющимися значениями.
    
    Параметры:
        df (pd.DataFrame): Исходный датафрейм для оптимизации памяти.
    
    Возвращает:
        pd.DataFrame: Тот же датафрейм с оптимизированными типами данных.
    
    Побочные эффекты:
        - Логирует размер до и после оптимизации, а также процент экономии памяти.
        - Модифицирует исходный датафрейм (изменяет типы данных колонок).
    
    Пример:
        df = pd.read_csv('data.csv')
        df = reduce_memory_usage(df)
        # Размер уменьшился с 72.65 MB до 5.89 MB (экономия 91.9%)
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
    
    logging.info(f"Размер после: {after:.2f} MB. Экономия: {(before-after)/before*100:.1f}%\n")
    return df

def calculate_mertics(model_name,  y_true, y_pred) -> list:
    """
    Вычисляет и логирует основные метрики качества бинарной классификации.
    
    Функция рассчитывает три ключевые метрики для оценки модели:
    - Precision (точность): доля объектов, названных положительными, которые действительно
      являются положительными. Отвечает на вопрос: "Насколько мы точны, когда говорим 'да'?"
    - Recall (полнота): доля реальных положительных объектов, которые модель смогла обнаружить.
      Отвечает на вопрос: "Какую долю положительных объектов мы нашли?"
    - F1-score: гармоническое среднее precision и recall, обеспечивающее баланс между ними.
    
    Параметры:
        model_name (str): Название модели для логирования (например, 'LogisticRegression').
        y_true (pd.Series): Истинные значения целевой переменной (0 или 1).
        y_pred (np.ndarray): Предсказания модели (0 или 1).
    
    Возвращает:
        list: Список из трёх значений [precision, recall, f1] в указанном порядке.
    
    Побочные эффекты:
        - Логирует значения метрик в формате: '{model_name} Precision = X.XX, Recall = X.XX, F1 = X.XX'
    
    Пример:
        metrics = calculate_metrics('KNN', y_valid, y_pred_knn)
        print(f"F1-score: {metrics[2]:.2f}")
    """
    
    precision = precision_score(y_true= y_true, y_pred= y_pred, average='binary')
    recall = recall_score(y_true= y_true, y_pred= y_pred, average='binary')
    f1 = f1_score(y_true= y_true, y_pred= y_pred, average='binary')
    
    logging.info(f'{model_name}\t Precison = {precision:.2f}, Recall = {recall:.2f}, F1 = {f1:.2f}')
    return [precision, recall, f1]

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


def calculate_precision_my(tp, fp):
    precision = tp / (tp + fp)
    return precision

def calculate_recall_my(tp, fn):
    recall = tp / (tp + fn)
    return recall

def calculate_f1_my(precision, recall):
    f1 = 2 * ( ( precision * recall ) / ( precision + recall ) )
    return f1

def calculate_auc_pr_my(y_true, y_proba):
    sorted_indices = np.argsort(y_proba)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    tp = 0
    fp = 0
    total_positives = np.sum(y_true == 1)
    
    ap_sum = 0
    num_positives = 0
    
    for i, label in enumerate(y_true_sorted):
        if label == 1:
            tp += 1
            num_positives += 1
            precision_at_i = tp / (tp + fp)
            ap_sum += precision_at_i
        else:
            fp += 1
    
    if num_positives == 0:
        return 0.0
    
    return ap_sum / total_positives
       
def calculate_metrics_my(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = calculate_precision_my(tp, fp)
    recall = calculate_recall_my(tp, fn)
    f1 = calculate_f1_my(precision, recall)
    auc_pr = calculate_auc_pr_my(y_true, y_proba)

    return precision, recall, f1, auc_pr