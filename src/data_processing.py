'''
Data Processing Module for Calorie Expenditure Analysis

This module contains all data cleaning, preprocessing, and feature engineering
functions for the fitness dataset analysis project.
'''

import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings

def load_data(file_path, dataset_type='synthetic_train'):
    '''
        Load the dataset into a dataframe, specifying the if the dataset comes from the synthetic or original version. 

        Args:
            filepath (str): path to csv file
            dataset_type (str): corresponding label for dataset (synthetic_train, synthetic_test, og_calories)

        Returns:
            Dataframe 
    '''
    try:
        df = pd.read_csv(file_path)
        print(f'Loaded {dataset_type} dataset: {df.shape}')
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f'Dataset not found at {file_path}')
    except Exception as e:
        raise Exception(f'Error loading dataset: {str(e)}')
    

def basic_data_info(df, dataset_name='Dataset'):
    '''
        Print a simple dataset overview 

        Args:
            df (DataFrame): dataset to analyze
            dataset_name (str): Name for clear display
    '''
    print(f'\n{dataset_name} Overview:')
    print("=" * 50)
    print(f'Shape: {df.shape}')
    print(f'\nData Types:\{df.dtypes}')
    print(f'\nMissing Values:\n{df.isnull().sum()}')
    print(f'\nBasic Stats:\n{df.describe()}')


def combine_datasets(synthetic_path, original_path):
    """
    load, clean, and combine datasets with proper tagging
    
    Returns:
        combined_df: Ready for feature engineering
    """

    # load datasets
    synthetic_df = load_data(synthetic_path, dataset_type='synthetic_train')
    og_df = load_data(original_path, dataset_type='og_all')
    
    # clean column names
    clean_column_names(synthetic_df)
    clean_column_names(og_df)
    
    # standardize column names
    og_df = og_df.rename(columns={'gender': 'sex'})
    og_df.drop(columns=['user_id'], inplace=True)
    
    # add dataset tags
    synthetic_df['tag'] = 'syn_train'
    og_df['tag'] = 'og_all'
    
    # combine
    combined_df = pd.concat([synthetic_df, og_df], ignore_index=True)
    
    print(f"✅ Combined datasets: {len(combined_df)} total samples")
    print(f"   • Synthetic: {len(synthetic_df)} samples")
    print(f"   • Original: {len(og_df)} samples")
    
    return combined_df


def kg_to_lbs(df):
    '''
        Convert the DataFrame's weight column units from kilograms to pounds

        Args:
            df (DataFrame): dataset
        
        Returns:
            df (DataFrame): dataset with converted values 
    '''
    df['weight'] = df['weight'] * 2.2046
    return df

def celsius_to_farhenheit(df):
    '''
        Convert the body temperature data from celsius to farhenheit

        Args:
            df (DataFrame): dataset
        
        Returns:
            df (DataFrame): dataset with converted values
    '''
    df['body_temp'] = (df['body_temp'] * 9/5) + 32
    return df

def cm_to_in(df):
    '''
        Convert centimeters to inches

        Args:
            df (DataFrame): dataset
        
        Returns:
            df (DataFrame): dataset with converted values
    '''

    df['height'] = df['height'] / 2.54
    return df

def clean_column_names(df):
    '''
        Standardize column names

        Args:
            df (DataFrame): dataset
        
        Returns:
            DataFrame
    '''
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)

    return df

def identify_numerical_columns(df, exclude_cols=None):
    '''
        Identify numerical columns 

        Args:
            df (DataFrame): dataset
            exclude_cols (optional list): Columns to exclude from numerical analysis

        Returns:
            list of numerical column names
    '''

    if exclude_cols is None:
        exclude_cols = ['user_id']
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

    return numerical_cols


def calc_num_bins(df):
    '''
        Use sturge's rule to calculate the number of bins when creating histograms

        Args:
            df (DataFrame): dataset
        
        Returns:
            number(int) of bins
    '''
    num_observations = df.shape[0]
    num_bins = int(math.log2(num_observations) + 1)
    
    return num_bins


def skewness(column):
    '''
        Determine the skewness of a given variable/column

        Args:
            column (pd.Series): numerical column/series to analyze
        
        Returns:
            measure of skewness (float) --> positive = right skew, negative = left skew
    '''
    return (3 * (column.mean() - column.median())) / np.std(column)


def kurtosis(column):
    '''
        Determine the skewness of a given variable/column

        Args:
            column (pd.Series): numerical column/series to analyze

        Returns:
            measure of kurtosis (float) --> 3 = normal, < 3 = leptokurtic, > 3 = platykurtic
    '''
    return stats.kurtosis(column, fisher=False)


def find_high_corr_pairs(df):
    '''
        Create a list of high correlation variable pairs (> 0.7) and a correlation matrix

        Args:
            df (DataFrame): dataset
        
        Returns:
            a correlation matrix object, a sorted list of high correlaion pairs
    '''

    correlation_matrix = df.select_dtypes(include=[np.number]).corr()

    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.7:
                high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], corr_val))
    
    high_corr_pairs.sort(key=lambda pair: abs(pair[2]), reverse=True)


    return correlation_matrix, high_corr_pairs


def target_variable_r_values(df):
    '''
        Create a list of r-values for the relationship between variables and the target variable

        Args:
            df (DataFrame): dataset containing calories column as target variable
        
        Returns:
            list of r values
    '''

    numerical_cols = identify_numerical_columns(df, exclude_cols=None)
    predictors = [col for col in numerical_cols if col != 'calories']
    corr_vals = []
    for i, var in enumerate(predictors):
        r = df[var].corr(df['calories'])
        corr_vals.append((var, r))

    corr_vals.sort(key= lambda x: x[1], reverse=True)

    return corr_vals



def detect_outliers_iqr(df, column):
    '''
        Use the IQR method to detect outliers for each variable

        Args:
            df (DataFrame): dataset
            column (str): column name to analyze for outliers
        
        Returns:
            tuple: (outliers_dataframe, lower_boundary_value, upper_boundary_value)
    '''
    q1 = df[column].quantile(.25)
    q3 = df[column].quantile(.75)

    iqr = q3 - q1
    upper_boundary = q3 + (1.5 * iqr)
    lower_boundary = q1 - (1.5 * iqr)

    outliers = df[(df[column] < lower_boundary) | (df[column] > upper_boundary)]
    return outliers, lower_boundary, upper_boundary


def gen_outlier_summary(df):
    '''
        Generate a summary of outlier information for each numerical column in the DataFrame
        
        Args:
            df (DataFrame): dataset to analyze
            
        Returns:
            dict: dictionary with outlier statistics for each column
                - count: number of outliers
                - lower_boundary: IQR lower bound
                - upper_boundary: IQR upper bound
    '''

    outlier_summery = {}
    numerical_cols = identify_numerical_columns(df, None)

    for col in numerical_cols:
        outliers, lower, upper = detect_outliers_iqr(df, col)
        outlier_summery[col] = {
            'count': len(outliers),
            'lower_boundary': lower,
            'upper_boundary': upper
        }

        print(f"{col}: {outlier_summery['count']} outliers, ({outlier_summery['count'] / len(df) * 100:.2f}%)")
        print(f'Bounds: [{lower:.2f}, {upper:.2f}]')
        print()
    
    return outlier_summery


def calculate_calorie_burn_rate(df):
    '''
        Calculate calorie burn rate (calories per minute) for outlier detection
        
        Args:
            df (DataFrame): dataset with calories and duration columns
            
        Returns:
            df (DataFrame): dataset with added calorie_burn_rate column
    '''
    df = df.copy()
    df['calorie_burn_rate'] = df['calories'] / df['duration']
    return df


def remove_calorie_rate_outliers_synthetic(df):
    '''
        Remove all calorie rate outliers from synthetic dataset (generation artifacts)
        
        Args:
            df (DataFrame): synthetic dataset with calorie_burn_rate column
            
        Returns:
            tuple: (cleaned_dataframe, removal_summary_dict)
    '''
    df = df.copy()
    
    # Get outliers using IQR method
    outliers, lower, upper = detect_outliers_iqr(df, 'calorie_burn_rate')
    
    # Remove all outliers
    df_clean = df[~df.index.isin(outliers.index)].copy()
    
    removal_summary = {
        'original_count': len(df),
        'outliers_removed': len(outliers),
        'final_count': len(df_clean),
        'percentage_removed': len(outliers) / len(df) * 100,
        'method': 'IQR_all_outliers'
    }
    
    print(f"Removed {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%) from synthetic dataset")
    
    return df_clean, removal_summary


def remove_calorie_rate_outliers_original(df, min_rate=2.0):
    '''
        Remove only unrealistically LOW calorie rates from original dataset
        
        Args:
            df (DataFrame): original dataset with calorie_burn_rate column
            min_rate (float): minimum realistic exercise calorie rate per minute
            
        Returns:
            tuple: (cleaned_dataframe, removal_summary_dict)
    '''
    df = df.copy()
    
    # count records below minimum rate
    low_outliers = df[df['calorie_burn_rate'] < min_rate]
    
    # keep only records with realistic calorie burn rates
    df_clean = df[df['calorie_burn_rate'] >= min_rate].copy()
    
    removal_summary = {
        'original_count': len(df),
        'outliers_removed': len(low_outliers),
        'final_count': len(df_clean),
        'percentage_removed': len(low_outliers) / len(df) * 100,
        'method': f'low_end_only_min_{min_rate}',
        'min_rate_threshold': min_rate
    }
    
    print(f"Removed {len(low_outliers)} low-end outliers ({len(low_outliers)/len(df)*100:.2f}%) from original dataset")
    
    return df_clean, removal_summary


def investigate_duration_calorie_relationship(df, n_extremes=10):
    '''
        Examine duration-calorie relationship to detect calculation artifacts
        
        Args:
            df (DataFrame): dataset with duration, calories, and calorie_burn_rate columns
            n_extremes (int): number of extreme values to display
            
        Returns:
            dict: analysis results with highest/lowest rates and statistics
    '''
    print('===== Duration v Calorie Range Analysis =====')
    
    # get highest and lowest calorie rates
    highest_rates = df.nlargest(n_extremes, 'calorie_burn_rate')[['duration', 'calories', 'calorie_burn_rate']]
    lowest_rates = df.nsmallest(n_extremes, 'calorie_burn_rate')[['duration', 'calories', 'calorie_burn_rate']]
    
    print(f"Highest {n_extremes} calorie rates (cal/min):")
    print(highest_rates)
    print(f"\nLowest {n_extremes} calorie rates (cal/min):")
    print(lowest_rates)
    
    # calculate statistics
    mean_rate = df['calorie_burn_rate'].mean()
    std_rate = df['calorie_burn_rate'].std()
    min_rate = df['calorie_burn_rate'].min()
    max_rate = df['calorie_burn_rate'].max()
    
    print(f"\nCalorie rate statistics:")
    print(f"Mean: {mean_rate:.2f} cal/min")
    print(f"Std: {std_rate:.2f} cal/min")
    print(f"Range: {min_rate:.2f} - {max_rate:.2f}")
    
    analysis_results = {
        'highest_rates': highest_rates,
        'lowest_rates': lowest_rates,
        'statistics': {
            'mean': mean_rate,
            'std': std_rate,
            'min': min_rate,
            'max': max_rate
        }
    }
    
    return analysis_results



def gen_body_temp_analysis(df):
    '''
        Generate a summary of the dataset's body temperature data against physiological norms

        Args:
            df (DataFrame): dataset containing 'body_temp' column in Fahrenheit
            ...
    '''
    print('===== Body Temperature Range Analysis =====')
    print(f"Dataset Range: {df['body_temp'].min():.1f}°F - {df['body_temp'].max():.1f}°F")
    print(f'Expected Range: 99.0°F - 101.0°F')
    print('Heat Exhaustion Range: 101.0°F - 104.0°F')

    normal_exercise_low = 99.0
    normal_exercise_high = 101.0
    heat_exhaustion_high = 104.0

    below_expected = (df['body_temp'] < normal_exercise_low).sum()
    normal_range = ((df['body_temp'] >= normal_exercise_low) & (df['body_temp'] < normal_exercise_high)).sum()
    above_normal = ((df['body_temp'] >= normal_exercise_high) & (df['body_temp'] < heat_exhaustion_high)).sum()
    dangerous = (df['body_temp'] > heat_exhaustion_high).sum()


    print(f'\nBelow Expected Exercise Range: (<{normal_exercise_low}°F): {below_expected:,} ({below_expected/len(df)*100:.2f}%)')
    print(f'Normal Expected Exercise Range: ({normal_exercise_low} - {normal_exercise_high}°F): {normal_range:,} ({normal_range/len(df)*100:.2f}%)')
    print(f'Elevated But Safe Range ({normal_exercise_high} - {heat_exhaustion_high}°F): {above_normal:,} ({above_normal/len(df)*100:.2f}%)')
    print(f'Dangerous Range (>{heat_exhaustion_high}°F): {dangerous:,} ({dangerous/len(df)*100:.2f}%)')



def categorize_bmi(df):
    conditions = [
        df['bmi'] < 18.5,
        (df['bmi'] >= 18.5) & (df['bmi'] < 25),
        (df['bmi'] >= 25) & (df['bmi'] < 30),
        (df['bmi'] >= 30) & (df['bmi'] < 35),
        (df['bmi'] >= 35) & (df['bmi'] < 40),
        df['bmi'] >= 40
    ]

    choices = [
        'underweight',
        'healthy',
        'overweight',
        'obesity1',
        'obesity2',
        'obesity3'
    ]
    
    df['bmi_category'] = np.select(conditions, choices, default='unknown')
    return df


'''
    BMR calculations using mifflin - st jeor formulas
        Men:
            10 * weight(kg) + 6.25 x height(cm) - 5 x age + 5
        Women:
            10 * weight(kg) + 6.25 x height(cm) - 5 x age - 161
'''

def calculate_bmr(df):
    conditions = [
        df['sex'] == 'male',
        df['sex'] == 'female'
    ]

    choices = [
        10 * df['weight'] + 6.25 * df['height'] - 5 * df['age'] + 5,
        10 * df['weight'] + 6.25 * df['height'] - 5 * df['age'] - 161
    ]
    
    df['bmr'] = pd.to_numeric(np.select(conditions, choices, default='unknown'), errors='coerce')
    return df


def calculate_metabolic_efficiency(df):
    df['met_efficiency'] = df['calories'] / df['bmr']

    return df



# body surface area using dubois formula

def calculate_bsa(df):
    df['bsa'] = 0.007184 * (df['height'] ** 0.725) * (df['weight'] ** 0.425)

    return df


def calculate_hr_percentage(df):

    theoretical_max = 220 - df['age']

    df['hr_percentage'] = df['heart_rate'] / theoretical_max

    return df


def categorize_hr_zone(df):
    conditions = [
        df['hr_percentage'] < 0.50,
        (df['hr_percentage'] >= .50) & (df['hr_percentage'] < .60),
        (df['hr_percentage'] >= .60) & (df['hr_percentage'] < .70),
        (df['hr_percentage'] >= .70) & (df['hr_percentage'] < .80),
        (df['hr_percentage'] >= .80) & (df['hr_percentage'] < .90),
        df['hr_percentage'] >= .90
    ]

    choices = [
        'Light',
        'Zone 1',
        'Zone 2',
        'Zone 3',
        'Zone 4',
        'Zone 5'
    ]

    df['hr_zone'] = np.select(conditions, choices, default='unknown')

    return df


def calculate_effort_score(df):
    
    df['effort_score'] = df['hr_percentage'] * df['duration']
    return df


def calculate_thermoregulatory_strain(df):

    df['thermo_strain'] = df['body_temp'] * df['duration']
    return df



def categorize_age(df):
    conditions = [
        df['age'] < 20,
        (df['age'] >= 20) & (df['age'] <= 30),
        (df['age'] > 30) & (df['age'] <= 40),
        (df['age'] > 40) & (df['age'] <= 50),
        (df['age'] > 50) & (df['age'] <= 60),
        (df['age'] > 60) & (df['age'] <= 70),
        df['age'] > 70
    ]

    choices = [
        'Under 20',
        '20s',
        '30s',
        '40s',
        '50s',
        '60s',
        '70s+'
    ]

    df['age_bin'] = np.select(conditions, choices, default='unknow')

    return df




def categorize_correlations(high_corr_df):
    """
    categorize correlations by type for better understanding
    """
    categories = {
        'Calculated from Base Features': [],
        'Interaction Terms': [],
        'Physiological Related': [],
        'Duration Related': [],
        'Other': []
    }
    
    for idx, row in high_corr_df.iterrows():
        var1, var2 = row['Variable 1'], row['Variable 2']
        pair = f"{var1} ↔ {var2}"
        corr = row['Correlation']
        
        # calculated features
        if any(calc in [var1, var2] for calc in ['bmi', 'bmr', 'bsa', 'hr_percentage']):
            categories['Calculated from Base Features'].append(f"{pair}: {corr:.3f}")
        
        # interaction terms
        elif any(inter in [var1, var2] for inter in ['effort_score', 'thermo_strain', 'met_efficiency']):
            categories['Interaction Terms'].append(f"{pair}: {corr:.3f}")
        
        # duration related
        elif 'duration' in [var1, var2]:
            categories['Duration Related'].append(f"{pair}: {corr:.3f}")
        
        # physiological
        elif any(phys in [var1, var2] for phys in ['height', 'weight', 'age', 'heart_rate', 'body_temp']):
            categories['Physiological Related'].append(f"{pair}: {corr:.3f}")
        
        else:
            categories['Other'].append(f"{pair}: {corr:.3f}")
    
    print("\nCORRELATION CATEGORIES:")
    print("=" * 60)
    
    for category, pairs in categories.items():
        if pairs:
            print(f"\n{category}:")
            print("-" * len(category))
            for pair in pairs:
                print(f"  • {pair}")