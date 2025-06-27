import os
import sys 
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import test_train_split


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import data_processing
from utils import (
    create_output_dir, save_model, evaluate_model, print_metrics, compare_scenarios, save_results
)

class CalorieModelingPipeline: 
    ''' 
    Pipeline for training and evaluating calorie prediction models
    '''

    def __init__(self, synthetic_path='../data/raw/synthetic_train.csv', original_path='../data/raw/og_calories.csv'):
        '''
        Initialize pipeline with data paths
        
        Args:
            synthetic_path: Path to synthetic training data
            original_path: Path to original calories data
        '''
        self.synthetic_path = synthetic_path 
        self.original_path = original_path

        # use the best model from optimization (03_modeling notebook)
        self.best_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.5, 
            'random_state': 42,
            'eval_metric': 'rmse'
        }

        self.top_features = [
            'effort_score', 'heart_rate', 'sex_encoded', 'met_efficiency', 'weight', 'bmr', 'thermo_strain', 'age', 'bsa', 'hr_percentage'
        ] 

        self.feature_config = {
            'numeric': ['age', 'height', 'weight', 'duration', 'heart_rate', 'body_temp',
                       'bmi', 'bmr', 'met_efficiency', 'bsa', 'hr_percentage', 
                       'effort_score', 'thermo_strain'],
            'categorical': ['sex', 'bmi_category', 'hr_zone', 'age_bin'],
            'target': 'calories'
        }

        self.data_loaded = False  
        self.results = {}

    def load_and_prepare_data(self):
        print('Loading and Preparing data...')
        print('=' * 60)

        combined_df = data_processing.combine_datasets(self.synthetic_path, self.original_path)

        combined_df = data_processing.kg_to_lbs(combined_df)
        combined_df = data_processing.celsius_to_farhenheit(combined_df)
        combined_df = data_processing.cm_to_in(combined_df)

        combined_df = data_processing.calculate_calorie_burn_rate(combined_df)

        #  separate by dataset types before outlier removal
        self.original_data = combined_df[combined_df['tag'] == 'og_all'].copy()
        self.synthetic_data = combined_df[combined_df['tag'] == 'syn_train'].copy()   

        print(f'Original data shape: {self.original_data.shape}')
        print(f'Synthetic data shape: {self.synthetic_data.shape}')

        print('\nRemoving outliers...')
        self.original_clean, _ = data_processing.remove_calorie_rate_outliers_original(self.original_data, min_rate=2.0)
        self.synthetic_clean, _ = data_processing.remove_calorie_rate_outliers_synthetic(self.synthetic_data)

        print(f'Original clean shape: {self.original_clean.shape}')
        print(f'Synthetic clean shape: {self.synthetic_clean.shape}')
        
        # split original data (70/15/15)
        print('\nSplitting original data...')
        self._split_original_data()

        print('\nApplying feature engineering...')
        self._apply_feature_engineering()

        print(f'\nPreparing features for modeling...')
        self._prepare_features()

        self.data_loaded = True
        print(f'\nData Prep Complete')

    def _split_original_data(self):
        original_features = self.original_clean.drop(columns=['tag'])

        # first split
        train_val, test = test_train_split(
            original_features, test_size=0.15, random_state=42, stratify=original_features['sex']
        )

        # second split
        val_proportion = 0.15 / (1 - 0.15)
        train, val = test_train_split(
            train_val, test_size=val_proportion, random_state=42, stratify=train_val['sex']
        )

        self.og_train = train
        self.og_val = val
        self.og_test = test

        print(f'Original splits - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}')

    def _apply_feature_engineering(self): 

        datasets_to_engineer = {
            'og_train': self.og_train,
            'og_val': self.og_val,
            'og_test': self.og_test,
            'synthetic_full': self.synthetic_clean.drop(columns=['tag'])
        }

        for name, df in datasets_to_engineer.items():
            df = data_processing.calculate_bmi(df)
            df = data_processing.categorize_bmi(df)
            df = data_processing.calculate_bmr(df)
            df = data_processing.calculate_bsa(df)
            df = data_processing.calculate_metabolic_efficiency(df)
            df = data_processing.calculate_hr_percentage(df)
            df = data_processing.categorize_hr_zone(df)
            df = data_processing.calculate_effort_score(df)
            df = data_processing.calculate_thermoregulatory_strain(df)
            df = data_processing.categorize_age(df)
            
            # store engineered dataset
            setattr(self, f'{name}_eng', df)

    def _prepare_features(self):
        self.encoders = {}
        self.scaler = StandardScaler() # scales by removing the mean and scaling to unit variance (x - mean)/stdev

        # fit encoders on training data
        for cat_feature in self.feature_config['categorical']:
            encoder = LabelEncoder()
            encoder.fit(self.og_train_eng[cat_feature])
            self.encoders[cat_feature] = encoder

        self.X_train, self.y_train = self._encode_and_scale(self.og_train_eng, fit_scaler=True)
        self.X_val, self.y_val = self._encode_and_scale(self.og_val_eng)
        self.X_test, self.y_test = self.encode_and_scale(self.og_test_eng)
        self.X_synthetic, self.y_synthetic = self.encode_and_scale(self.synthetic_full_eng)

        #  extract top feature indices 
        all_feature_names = self._get_all_feature_names()
        self.top_feature_indices = [all_feature_names.index(f) for f in self.top_features]

        print(f'Feature prep complete. Total features: {len(all_feature_names)}')
        print(f'Using top {len(self.top_features)} features for modeling')

    def _encode_and_scale(self, df, fit_scaler=False): 
        df_processed = df.copy()

        for cat_feature in self.feature_config['categorical']:
            if cat_feature in df_processed.columns:
                df_processed[f'{cat_feature}_encoded'] = self.encoders[cat_feature].transform(df_processed[cat_feature]) 

        # prep feature matrix
        feature_columns = []

        for cat_feature in self.feature_config['categorical']:
            if cat_feature in df_processed.columns:
                feature_columns.append(f'{cat_feature}_encoded')

        feature_columns.extend(self.feature_config['numeric'])

        # extract features and targets
        X = df_processed[feature_columns].values
        y = df_processed[self.feature_config['target']].values

        # scale numeric features
        numeric_start_idx = len(self.feature_config['categorical'])
        X_numeric = X[:, numeric_start_idx:] 

        if fit_scaler:
            X_numeric_scaled = self.scaler.fit_transform(X_numeric)
        else:
            X_numeric_scaled = self.scaler.transform(X_numeric)

        if self.feature_config['categorical']:
            X_categorical = X[:, :numeric_start_idx]
            X = np.column_stack([X_categorical, X_numeric_scaled])
        else:
            X = X_numeric_scaled

        return X, y
    

    def _get_all_feature_names(self):
        feature_names = []

        # encoded categorical features 
        for cat_feature in self.feature_config['categorical']:
            feature_names.append(f'{cat_feature}_encoded')

        feature_names.extend(self.feature_config['numeric'])

        return feature_names
    
    def run_scenario_1(self, save_model_flag=False, output_dir=None): 
        ''' 
        Train on original data only
        '''
        print('\n' + '=' * 60)
        print('SCENARIO 1: Original Data Only')
        print('=' * 60)

        # extract top features 
        X_train_top = self.X_train[:, self.top_feature_indices]
        X_val_top = self.X_val[:, self.top_feature_indices]
        X_test_top = self.X_test[:, self.top_feature_indices]

        print(f'Training data shape: {X_train_top.shape}')

        # train
        model = xgb.XGBRegressor(**self.best_params)
        model.fit(X_train_top, self.y_train)

        # eval 
        train_pred = model.predict(X_train_top)
        val_pred = model.predict(X_val_top)
        test_pred = model.predict(X_test_top)

        train_metrics = evaluate_model(self.y_train, train_pred, 'Scenario 1 - Train')
        val_metrics = evaluate_model(self.y_val, val_pred, 'Scenario 1 - Val') 
        test_metrics = evaluate_model(self.y_test, test_pred, 'Scenario 1 - Test')

        print_metrics(train_metrics, 'Training Set') 
        print_metrics(val_metrics, 'Validation Set') 
        print_metrics(test_metrics, 'Test Set')


        # optional save model
        if save_model_flag and output_dir:
            metadata = {
                'scenario': 1,
                'training_data': 'original_only',
                'n_samples': len(self.y_train),
                'features': self.top_features,
                'best_params': self.best_params,
                'metrics': {
                    'train': train_metrics,
                    'val': val_metrics,
                    'test': test_metrics
                }
            }
            save_model(model, 'xgboost', 1, output_dir, metadata)  

        self.results['Scenario 1: Original Only'] = {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
            'model': model
        }

        return model
    
    def run_scenario_2(self, save_model_flag=False, output_dir=None):
        '''Scenario 2: Train on original + synthetic data'''
        print('\n' + '=' * 60)
        print('SCENARIO 2: Original + Synthetic Data')
        print('=' * 60)
        
        # combine training data
        X_combined = np.concatenate([self.X_train, self.X_synthetic])
        y_combined = np.concatenate([self.y_train, self.y_synthetic])
        
        X_combined_top = X_combined[:, self.top_feature_indices]
        X_val_top = self.X_val[:, self.top_feature_indices]
        X_test_top = self.X_test[:, self.top_feature_indices]
        
        print(f'Combined training data shape: {X_combined_top.shape}')
        print(f'  - Original samples: {len(self.y_train)}')
        print(f'  - Synthetic samples: {len(self.y_synthetic)}')
        
        # train model
        model = xgb.XGBRegressor(**self.best_params)
        model.fit(X_combined_top, y_combined)
        
        # evaluate on original data only
        train_pred = model.predict(self.X_train[:, self.top_feature_indices])
        val_pred = model.predict(X_val_top)
        test_pred = model.predict(X_test_top)
        
        train_metrics = evaluate_model(self.y_train, train_pred, 'Scenario 2 - Train (Original)')
        val_metrics = evaluate_model(self.y_val, val_pred, 'Scenario 2 - Val')
        test_metrics = evaluate_model(self.y_test, test_pred, 'Scenario 2 - Test')
        
        print_metrics(train_metrics, 'Training Set (Original Data Only)')
        print_metrics(val_metrics, 'Validation Set')
        print_metrics(test_metrics, 'Test Set')
    
        if save_model_flag and output_dir:
            metadata = {
                'scenario': 2,
                'training_data': 'original_plus_synthetic',
                'n_samples_original': len(self.y_train),
                'n_samples_synthetic': len(self.y_synthetic),
                'n_samples_total': len(y_combined),
                'features': self.top_features,
                'best_params': self.best_params,
                'metrics': {
                    'train': train_metrics,
                    'val': val_metrics,
                    'test': test_metrics
                }
            }
            save_model(model, 'xgboost', 2, output_dir, metadata)
        
        self.results['Scenario 2: Original + Synthetic'] = {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
            'model': model
        }
        
        return model


    def run_scenario_3(self, save_model_flag=False, output_dir=None):
        '''Scenario 3: Train on synthetic data only'''
        print('\n' + '=' * 60)
        print('SCENARIO 3: Synthetic Data Only')
        print('=' * 60)

        X_synthetic_top = self.X_synthetic[:, self.top_feature_indices]
        X_val_top = self.X_synthetic[:, self.top_feature_indices]
        X_test_top = self.X_synthetic[:, self.top_feature_indices]

        print(f'Synthetic training data shape: {X_synthetic_top.shape}') 

        model = xgb.XGBRegressor(**self.best_params)
        model.fit(X_synthetic_top, self.y_synthetic)

        # evaluate on a sample of the synthetic data
        synthetic_sample_size = min(len(self.y_train), len(self.y_synthetic))
        sample_indices = np.random.choice(len(self.y_synthetic), synthetic_sample_size, replace=False)

        train_pred = model.predict(X_synthetic_top[sample_indices]) 
        val_pred = model.predict(X_val_top)
        test_pred = model.predict(X_test_top)

        train_metrics = evaluate_model(
            self.y_synthetic[sample_indices],
            train_pred,
            'Scenario 3 - Train (Synthetic Sample)'
        )

        val_metrics = evaluate_model(self.y_val, val_pred, 'Scenario 3 - Val')
        test_metrics = evaluate_model(self.y_test, test_pred, 'Scenario 3 - Test')

        if save_model_flag and output_dir:
            metadata = {
                'scenario': 3,
                'training_data': 'synthetic_only',
                'n_samples': len(self.y_synthetic),
                'features': self.top_features,
                'best_params': self.best_params,
                'metrics': {
                    'train': train_metrics,
                    'val': val_metrics,
                    'test': test_metrics
                }
            }
            save_model(model, 'xgboost', 3, output_dir, metadata)
        
        self.results['Scenario 3: Synthetic Only'] = {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics,
            'model': model
        }
        
        return model
    

    def run_all_scenarios(self, save_models=False):
        if not self.data_loaded:
            self.load_and_prepare_data()

        output_dir = None

        if save_models:
            output_dir = create_output_dir()
            print(f'\nOutput directory: {output_dir}')

        self.run_scenario_1(save_models, output_dir)
        self.run_scenario_2(save_models, output_dir)
        self.run_scenario_3(save_models, output_dir)

        compare_scenarios(self.results)

        if save_models:
            save_results(self.results, output_dir)

        return self.results