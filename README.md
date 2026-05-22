#  ML Projects from School 21

Портфолио проектов по машинному обучению, выполненных в рамках программы [School 21](https://21-school.ru/). В репозитории собраны ноутбуки с полным циклом разработки: от предобработки данных и feature engineering до обучения, регуляризации и оценки моделей.

##  Структура проектов

### 1. ML1: Introduction to Machine Learning
- **Задача:** Прогнозирование стоимости аренды квартир в Нью-Йорке (Kaggle: NYC Apartment Listings)
- **Что реализовано:** 
  - Парсинг и очистка текстовых признаков (`Features`)
  - Генерация бинарных фичей, работа с выбросами
  - Линейная регрессия (аналитическое решение, SGD, Batch/Mini-batch)
  - Ridge, Lasso, ElasticNet, сравнение метрик (MAE, RMSE, R²)
- 📓 [Ноутбук](ML1_Introduction.ID_1254798-1/src/ml1.ipynb)

### 2. ML2: Supervised Learning
- **Задача:** Продвинутая регрессия с полиномиальными признаками и регуляризацией
- **Что реализовано:**
  - Генерация полиномиальных фичей (degree=10)
  - Борьба с переобучением через L1/L2 регуляризацию
  - Нормализация признаков (MinMax, Standard)
  - Анализ стабильности моделей и подбор гиперпараметров
-  [Ноутбук](ML2_Supervised_learning.ID_1254799-1/src/ml2.ipynb)

##  Как запустить проекты

### 1. Скачайте данные
Датасеты доступны на [Kaggle: Two Sigma Connect](https://www.kaggle.com/competitions/two-sigma-connect-rental-listing-inquiries/data).  
Поместите файлы `train.json` и `test.json` в соответствующие папки `data/` перед запуском ноутбуков.

### 2. Создайте виртуальное окружение
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows