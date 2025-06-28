import os 
import json 
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
sns.set_palette('husl')

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12 
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def create_all_visualizations(results, output_dir):
    '''
    Create all visualizations for the synthetic data analysis
    
    Args:
        results: Dictionary containing results from all scenarios
        output_dir: Directory to save visualizations
    '''
    print('Creating visualizations...')

    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    has_predictions = any('predictions' in scenario_data for scenario_data in results.values())

    create_performance_comparison_bar(results, vis_dir)
    create_overfitting_analysis(results, vis_dir)
    create_metric_radar_chart(results, vis_dir)
    create_scenario_comparison_table(results, vis_dir)
    create_performance_improvement_waterfall(results, vis_dir)
    create_synthetic_impact_analysis(results, vis_dir)

    if has_predictions:
        create_prediction_error_distribution(results, vis_dir) 
        create_residual_plots(results, vis_dir)

    create_summary_report(results, vis_dir)

    print(f'All visualizations saved to {vis_dir}')


def create_performance_comparison_bar(results, vis_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
     
    scenarios = list(results.keys)
    scenario_labels = ['Original Only', 'Original + Synthetic', 'Synthetic Only'] 

    metrics_data = {
        'RMSLE': {'val': [], 'test': []},
        'MAE': {'val': [], 'test': []},
        'R²': {'val': [], 'test': []},
        'MAPE': {'val': [], 'test': []}
    }

    for scenario in scenarios:
        for metric in metrics_data:
            metric_lower = metric.lower() if metric != 'R²' else 'r2' 
            metrics_data[metric]['val'].append(results[scenario]['val'][metric_lower])
            metrics_data[metric]['test'].append(results[scenario]['test'][metric_lower])

    for idx, (metric, data) in enumerate(metrics_data.items()):
        ax = axes[idx // 2, idx % 2]

        x = np.arange(len(scenarios))
        width = 0.35 

        bars1 = ax.bar(x - width/2, data['val'], width, label='Validation', alpha=0.8)
        bars2 = ax.bar(x + width/2, data['test'], width, label='Test', alpha=0.8)

        # val labels 
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                format_str = '.4f' if metric == 'RMSLE' else '.3f' if metric == 'R²' else '.1f'
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:{format_str}}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Scenario')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison Across Scenarios')
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels, rotation=15, ha='right')
        ax.legend()

        # baseline reference 
        if metric != 'R²':
            baseline = data['val'][0]
            ax.axhline(y=baseline, color='red', linestyle='--', alpha=0.5, label='Baseline') 

    plt.suptitle('Performance Metrics Comparison Across All Scenarios')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'performance_comparison_detailed.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(' - Detailed performance comparison created')


def create_overfitting_analysis(results, vis_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    scenarios = list(results.keys()) 
    scenario_labels = ['Original\nOnly', 'Original  +\nSynthetic', 'Synthetic\nOnly'] 

    train_rmsle = [results[s]['train']['rmsle'] for s in scenarios]
    val_rmsle = [results[s]['val']['rmsle'] for s in scenarios]
    test_rmsle = [results[s]['test']['rmsle'] for s in scenarios]

    # plot 1: train vs val vs test
    x = np.arange(len(scenarios))
    width = 0.25

    bars1 = ax1.bar(x - width, train_rmsle, width, label='Train', alpha=0.8)
    bars2 = ax1.bar(x, val_rmsle, width, label='Validation', alpha=0.8) 
    bars3 = ax1.bar(x + width, test_rmsle, width, label='Test', alpha=0.8)

    ax1.set_xlabel('Scenario')
    ax1.set_ylabel('RMSLE')
    ax1.set_title('Train vs Validation vs Test RMSLE')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_labels)
    ax1.legend()


    # value annotations
    for bars in [bars1, bars2, bars3]: 
        for bar in bars:
            height = bar.get_height() 
            ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', va='bottom', fontsize=8, rotation=90)

    # plot 2: overfitting gaps
    val_gaps = [v - t for t, v in zip(train_rmsle, val_rmsle)]
    test_gaps = [test - t for t, test in zip(train_rmsle, test_rmsle)]

    bars4 = ax2.bar(x - width/2, val_gaps, width, label='Val - Train', alpha=0.8, color='orange')
    bars5 = ax2.bar(x + width/2, test_gaps, width, label='Test - Train', alpha=0.8, color='red')
    
    ax2.set_xlabel('Scenario')
    ax2.set_ylabel('RMSLE Gap')
    ax2.set_title('Overfitting Analysis (Gap from Training)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenario_labels)
    ax2.legend()
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # value annotations
    for bars in [bars4, bars5]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
            
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'overfitting_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(' - Overfitting analysis created')

def create_metric_radar_chart(results, vis_dir):
    scenarios = list(results.keys())
    
    # normalized metrics for comparison
    metrics = ['RMSLE', 'MAE', 'R²', 'MAPE', 'Overfitting'] 

    data = []
    for scenario in scenarios: 
        scenario_metrics = []

        rmsle = results[scenario]['val']['rmsle']
        mae = results[scenario]['val']['mae']
        r2 = results[scenario]['val']['r2']
        mape = results[scenario]['val']['mape']
        overfitting = results[scenario]['val']['rmsle'] - results[scenario]['train']['rmsle']

        # normalize metrics to a 0-1 scale (1 is best), invert error metrics so lower is better
        # assumptions are based of off exploring in modeling notebook 
        scenario_metrics.append(1 - min(rmsle / 0.2, 1)) # assuming max rmsle of 0.2
        scenario_metrics.append(1 - min(mae / 50, 1)) # assume max mae of 50 
        scenario_metrics.append(r2) # already scaled 0-1
        scenario_metrics.append(1 - min(mape / 10, 1)) # assume max mape of 10%
        scenario_metrics.append(1 - min(overfitting / 0.05, 1)) # assume max overfit of 0.05

        data.append(scenario_metrics)

    
    # create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist() 
    data = [d + [d[0]] for d in data] # complete the circle
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    colors = ['blue', 'green', 'red']
    for i, (scenario_data, scenario_name) in enumerate(zip(data, scenarios)): 
        ax.plot(angles, scenario_data, 'o-', linewidth=2, color=colors[i], label=scenario_name.replace('Scenario ', 'S'))
        ax.fill(angles, scenario_data, alpha=0.15, color=colors[i]) 

    # Customize chart
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.title('Multi-Metric Performance Comparison\n(Higher values are better)', pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'metric_radar_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print('  - Radar chart created')




