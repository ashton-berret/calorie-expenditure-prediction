''' 
    Utility functions for model training, evaluation, and persistance
'''

import os
import json
import pickle 
import pandas as pd
import numpy as np 
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def create_output_dir(base_dir='../outputs'):
    ''' 
    Create timestamped output dir for saving results
    '''
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    return output_dir


def save_model(model, model_name, scenario, output_dir, metadata=None):
    ''' 
    Save trained model with metadata
    
    Args:
        model: Trained model object
        model_name: Name of the model (e.g., 'xgboost')
        scenario: Scenario number (1, 2, or 3)
        output_dir: Directory to save the model
        metadata: Additional metadata to save with model
    '''
    model_filename = f'{model_name}_scenario_{scenario}.pkl'
    model_path = os.path.join(output_dir, model_filename)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


    if metadata: 
        metadata['model_path'] = model_path
        metadata['saved_at'] = datetime.now().isoformat()
        metadata_path = os.path.join(output_dir, f'{model_name}_scenario_{scenario}_metadata.json')
        with open(metadata_path, 'w') as f: 
            json.dump(metadata, f, indent=2)

    print(f'Model Saved: {model_path}')
    return model_path


def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def rmsle(y_true, y_pred):
    # handle potential negative predictions
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

def evaluate_model(y_true, y_pred, model_name='Model'):
    '''
    Returns dictionary of evaluation metrics 
    '''
    y_pred = np.maximum(y_pred, 0)

    metrics = {
        'model_name': model_name,
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'rmsle': rmsle(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        'mean_pred': np.mean(y_pred),
        'mean_actual': np.mean(y_true),
        'std_pred': np.std(y_pred),
        'std_actual': np.std(y_true)
    }

    return metrics

def print_metrics(metrics, dataset_name=''):
    print(f'\n{dataset_name} Evaluation:')
    print('-' * 40)
    print(f'RMSLE: {metrics["rmsle"]:.4f}')
    print(f'RMSE: {metrics["rmse"]:.4f}')
    print(f'MAE: {metrics["mae"]:.4f}')
    print(f'R²: {metrics["r2"]:.4f}')
    print(f'MAPE: {metrics["mape"]:.2f}%')
    print(f'Mean Pred vs Actual: {metrics["mean_pred"]:.2f} vs {metrics["mean_actual"]:.2f}')

def compare_scenarios(results_dict):
    '''
    Generate comparison report for multiple scenarios
    
    Args:
        results_dict: Dictionary with scenario names as keys and metrics as values
    '''
    print('\n' + '=' * 80)
    print('SCENARIO COMPARISON REPORT')
    print('=' * 80)
    
    # create comparison dataframe
    comparison_data = []
    for scenario, metrics in results_dict.items():
        row = {
            'Scenario': scenario,
            'Train RMSLE': metrics['train']['rmsle'],
            'Val RMSLE': metrics['val']['rmsle'],
            'Test RMSLE': metrics.get('test', {}).get('rmsle', None),
            'Val R²': metrics['val']['r2'],
            'Overfitting': metrics['val']['rmsle'] - metrics['train']['rmsle']
        }
        comparison_data.append(row)
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # print formatted table
    print('\nPerformance Summary:')
    print('-' * 80)
    print(f'{"Scenario":<30} {"Train RMSLE":<12} {"Val RMSLE":<12} {"Test RMSLE":<12} {"Val R²":<10} {"Overfit":<10}')
    print('-' * 80)
    
    for _, row in df_comparison.iterrows():
        test_rmsle = f'{row["Test RMSLE"]:.4f}' if row['Test RMSLE'] is not None else 'N/A'
        print(f'{row["Scenario"]:<30} {row["Train RMSLE"]:<12.4f} {row["Val RMSLE"]:<12.4f} {test_rmsle:<12} {row["Val R²"]:<10.4f} {row["Overfitting"]:<10.4f}')
    
    best_scenario = df_comparison.loc[df_comparison['Val RMSLE'].idxmin(), 'Scenario']
    best_val_rmsle = df_comparison['Val RMSLE'].min()
    
    print(f'\nBest Scenario: {best_scenario} (Val RMSLE: {best_val_rmsle:.4f})')
    
    baseline_val_rmsle = df_comparison[df_comparison['Scenario'].str.contains('Original Only')]['Val RMSLE'].values[0]
    
    print('\nComparison to Baseline (Original Only):')
    print('-' * 50)
    for _, row in df_comparison.iterrows():
        if 'Original Only' not in row['Scenario']:
            diff = row['Val RMSLE'] - baseline_val_rmsle
            pct_change = (diff / baseline_val_rmsle) * 100
            direction = 'improvement' if diff < 0 else 'degradation'
            print(f'{row["Scenario"]}: {abs(pct_change):.2f}% {direction}')
    
    return df_comparison

def save_results(results_dict, output_dir):
    # save detailed metrics as JSON

    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    comparison_df = compare_scenarios(results_dict)
    comparison_path = os.path.join(output_dir, 'comparison.csv')
    comparison_df.to_csv(comparison_path, index=False)

    print(f'\nResults saved to: {output_dir}')
    return results_path, comparison_path