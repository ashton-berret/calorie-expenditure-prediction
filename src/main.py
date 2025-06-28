import os
import sys
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modeling import CalorieModelingPipeline
from utils import create_output_dir, save_results

def print_header():
   print('\n' + '=' * 80)
   print('CALORIE PREDICTION: SYNTHETIC DATA UTILITY ANALYSIS')
   print('Research Question: Does synthetic data augmentation improve prediction?')
   print('=' * 80)
   print(f'Started at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


def print_menu():
    print('\n' + '-' * 50)
    print('MAIN MENU')
    print('-' * 50)
    print('1. Run Experiments Only')
    print('2. Generate Visualizations Only (requires existing results)')
    print('3. Run Full Pipeline (Experiments + Visualizations)')
    print('4. Quick Test (Scenario 1 only)')
    print('5. Exit')
    print('-' * 50)


def run_experiments(save_models=True):
    print('\n' + '=' * 60)
    print('RUNNING MODELING EXPERIMENTS')
    print('=' * 60)

    pipeline = CalorieModelingPipeline()

    pipeline.load_and_prepare_data()

    output_dir = create_output_dir() if save_models else None
    if output_dir:
        print(f'\nResults will be saved to {output_dir}')

    results = pipeline.run_all_scenarios(save_models)

    return results, output_dir

def generate_visualizations(results=None, output_dir=None):
    '''Generate all visualizations'''
    print('\n' + '=' * 60)
    print('GENERATING VISUALIZATIONS')
    print('=' * 60)

    try:
        # Try to import visualization module
        import visualization

        if results is None:
            # Load results from file
            if output_dir is None:
                # Find most recent results
                base_dir = '../outputs'
                if os.path.exists(base_dir):
                    dirs = [d for d in os.listdir(base_dir) if d.startswith('run_')]
                    if dirs:
                        dirs.sort()
                        output_dir = os.path.join(base_dir, dirs[-1])
                        print(f'Loading results from: {output_dir}')
                    else:
                        print('No previous results found. Please run experiments first.')
                        return
                else:
                    print('No outputs directory found. Please run experiments first.')
                    return

            # Load results from output directory
            import json
            results_path = os.path.join(output_dir, 'results.json')
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    results = json.load(f)
            else:
                print(f'Results file not found: {results_path}')
                return

        # Generate visualizations
        visualization.create_all_visualizations(results, output_dir)
        print(f'\nVisualizations saved to: {output_dir}')

    except ImportError as e:
        print(f'Error importing visualization module: {e}')
        print('Please ensure visualization.py is in the src directory.')
        print('\nCurrent working directory:', os.getcwd())
        print('Files in current directory:', os.listdir('.'))
    except Exception as e:
        print(f'Error generating visualizations: {e}')
        import traceback
        traceback.print_exc()


def run_quick_test():
    print('\n' + '=' * 60)
    print('QUICK TEST: Scenario 1 Only')
    print('=' * 60)

    pipeline = CalorieModelingPipeline()
    pipeline.load_and_prepare_data()

    output_dir = create_output_dir()
    print(f'\nResults will be saved to: {output_dir}')

    model = pipeline.run_scenario_1(save_model_flag=True, output_dir=output_dir)

    save_results(pipeline.results, output_dir)

    print('\nQuick test complete!')
    return pipeline.results, output_dir


def run_full_pipeline():
    print('\n' + '=' * 60)
    print('RUNNING FULL PIPELINE')
    print('=' * 60)

    results, output_dir = run_experiments(save_models=True)

    generate_visualizations(results, output_dir)

    print('\n' + '=' * 60)
    print('FULL PIPELINE COMPLETE')
    print('=' * 60)
    print(f'All results saved to: {output_dir}')

    return results, output_dir


def main():
    print_header()

    while True:
        print_menu()

        choice = input('\nEnter your choice (1-5): ').strip()

        if choice == '5':
            print('\nExiting application...')
            break

        elif choice == '1':
            # run experiments only
            save_models = input('\nSave trained models? (y/n): ').strip().lower() == 'y'
            results, output_dir = run_experiments(save_models)

            if output_dir:
                print(f'\nExperiment results saved to: {output_dir}')
                view_results = input('View results summary? (y/n): ').strip().lower() == 'y'
                if view_results:
                    from utils import compare_scenarios
                    compare_scenarios(results)

        elif choice == '2':
            # generate visualizations only
            use_latest = input('\nUse latest results? (y/n): ').strip().lower() == 'y'

            if use_latest:
                generate_visualizations()
            else:
                output_dir = input('Enter path to results directory: ').strip()
                if os.path.exists(output_dir):
                    generate_visualizations(output_dir=output_dir)
                else:
                    print(f'Directory not found: {output_dir}')

        elif choice == '3':
            confirm = input('\nThis will run all experiments and generate visualizations. Continue? (y/n): ').strip().lower()
            if confirm == 'y':
                results, output_dir = run_full_pipeline()

        elif choice == '4':
            results, output_dir = run_quick_test()

        else:
            print('Invalid choice. Please try again.')
            continue

        continue_choice = input('\nReturn to main menu? (y/n): ').strip().lower()
        if continue_choice != 'y':
            print('\nExiting application...')
            break

    print(f'\nCompleted at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')



if __name__ == '__main__':
    main()
