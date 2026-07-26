# ML Projects from School 21

Портфолио проектов по машинному обучению и анализу данных, выполненных в рамках программы [School 21](https://21-school.ru/). В репозитории собраны ноутбуки с полным циклом разработки: от предобработки данных и feature engineering до обучения, регуляризации и оценки моделей.

## Структура проектов

### 1. ML1: Introduction to Machine Learning
- **Задача:** Прогнозирование стоимости аренды квартир в Нью-Йорке (Kaggle: Two Sigma Connect)
- **Что реализовано:** 
  - Парсинг и очистка текстовых признаков (`Features`)
  - Генерация бинарных фичей, работа с выбросами
  - Линейная регрессия (аналитическое решение, SGD, Batch/Mini-batch)
  - Ridge, Lasso, ElasticNet, сравнение метрик (MAE, RMSE, R²)
- [Ноутбук](ML1_Introduction.ID_1254798-1/src/ml1.ipynb)

### 2. ML2: Supervised Learning
- **Задача:** Продвинутая регрессия с полиномиальными признаками и регуляризацией
- **Что реализовано:**
  - Генерация полиномиальных фичей (degree=10)
  - Борьба с переобучением через L1/L2 регуляризацию
  - Нормализация признаков (MinMax, Standard)
  - Анализ стабильности моделей и подбор гиперпараметров
- [Ноутбук](ML2_Supervised_learning.ID_1254799-1/src/ml2.ipynb)

### 3. ML3: Validation
- **Задача:** Валидация моделей машинного обучения и надежная оценка их качества
- **Что реализовано:**
  - Реализация методов кросс-валидации (K-Fold, Stratified K-Fold)
  - Подбор гиперпараметров с помощью Grid Search и Random Search
  - Построение и анализ кривых обучения (learning curves) для диагностики переобучения
- [Ноутбук](ML3_Validation.ID_1254800-1/src/ml_3.ipynb)

### 4. ML4: Classification Problems
- **Задача:** Бинарная классификация для выявления дефектных автомобилей (Kaggle: Don't Get Kicked)
- **Что реализовано:**
  - Строгий временной сплит данных (temporal split) для предотвращения утечки данных
  - Реализация с нуля Logistic Regression (SGD), GaussianNB и KNN
  - Генерация нелинейных признаков и групповых агрегаций
  - Отбор признаков через L1-регуляризацию и расчет кастомных метрик (Gini, ROC-AUC, AUC-PR)
- [Ноутбук](ML4_Classification_problems.ID_1254802-1/src/ml_4.ipynb)

### 5. ML5: Decision Trees
- **Задача:** Изучение и реализация деревьев решений и их ансамблей (Kaggle: Don't Get Kicked)
- **Что реализовано:**
  - Реализация с нуля DecisionTreeClassifier, RandomForestClassifier и GBDTClassifier (с расчетом градиентов BCE)
  - Применение и тонкая настройка LightGBM, CatBoost, XGBoost с помощью Optuna (Bayesian optimization)
  - Комплексный анализ лучшей модели: ROC-кривые, Feature Importance, диагностика переобучения
- [Ноутбук](ML5_Decision_trees_ID_1254804-1/src/ml_5.ipynb)

### 6. CT00: Career Track
- **Задача:** Карьерное планирование и постановка профессиональных целей
- **Что реализовано:**
  - Диагностика текущего профессионального состояния с помощью методики Career Balance Wheel
  - Определение целевого состояния и ключевых метрик успеха на 2-3 и 5 лет
  - Разработка пошагового плана развития (Action Steps) для перехода от студента School 21 до Middle/Senior ML Engineer в Сбере
- [Документ](CT00_ID_1576118-1/src/career_balance_wheel_and_goals_timetilne.pdf)

---

## Как запустить проекты

### 1. Скачайте данные
Проекты используют два основных датасета с Kaggle:
1. **ML1, ML2:** [Two Sigma Connect: Rental Listing Inquiries](https://www.kaggle.com/competitions/two-sigma-connect-rental-listing-inquiries/data)  
   *(Поместите файлы `train.json` и `test.json` в папку `data/` соответствующего проекта)*
2. **ML4, ML5:** [Don't Get Kicked!](https://www.kaggle.com/competitions/DontGetKicked/data)  
   *(Поместите файл `training.csv` в папку `data/` соответствующего проекта)*

### 2. Создайте виртуальное окружение

### 2. Создайте виртуальное окружение
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
