import optuna
import logging
from functions import show_results

class ModelFineTuner:
    """Класс для fine-tune моделей через Optuna."""
    
    def __init__(self, X_train, y_train, X_valid_inner, y_valid_inner, X_test_inner, y_test_inner):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid_inner = X_valid_inner
        self.y_valid_inner = y_valid_inner
        self.X_test_inner = X_test_inner
        self.y_test_inner = y_test_inner
    
    def fine_tune(self, model_name, base_model, objective, build_best_model, n_trials=50):
        logging.info(f'{model_name}: Fine-tune гиперпараметров')
        print(f'{model_name}: Fine-tune гиперпараметров')
        
        base_gini = show_results(
            base_model, self.X_test_inner, self.y_test_inner, verbose=False
        )
        logging.info(f'Gini базовой модели : {base_gini:.4f}')
        print(f'Gini базовой модели: {base_gini:.4f}')
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=21)
        
        def objective_wrapper(trial):# делаем замыкание, потому что глобальные переменные это плохо
            return objective(trial, self.X_train, self.y_train, 
                           self.X_valid_inner, self.y_valid_inner)
        
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective_wrapper, n_trials=n_trials, show_progress_bar=False)
        
        logging.info(f'Лучший Gini: {study.best_value:.4f}')
        logging.info(f'Лучшие параметры: {study.best_params}')
        print(f"Лучший Gini: {study.best_value:.4f}")
        print(f"Лучшие параметры: {study.best_params}")
        
        best_model = build_best_model(study.best_params)
        best_model.fit(self.X_train, self.y_train)
        final_gini = show_results(
            best_model, self.X_test_inner, self.y_test_inner, verbose=False
        )
        logging.info(f'Gini лучшей модели: {final_gini:.4f}')
        print(f'Gini лучшей модели: {final_gini:.4f}\n')
        
        improvement = (final_gini - base_gini) / base_gini * 100
        logging.info(f'Улучшение: +{improvement:.1f}%\n')
        
        return study.best_value, study.best_params, best_model, base_gini, final_gini