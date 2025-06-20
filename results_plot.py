import os
import numpy as np
import matplotlib.pyplot as plt 

# --- Constants ---
DEFAULT_OUTPUT_DIR = 'output_plots' 
DEFAULT_DPI = 300

def save_plot(fig, path, output_dir=DEFAULT_OUTPUT_DIR, dpi=DEFAULT_DPI):
    ''' Saves matplotlib figure to a specified directory. '''
    try:
        os.makedirs(output_dir, exist_ok=True) 
        full_path = os.path.join(output_dir, path)
        fig.savefig(full_path, dpi=dpi)
        print(f"Plot saved to {full_path}")
    except OSError as e:
        print(f"Error saving plot to {full_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving the plot: {e}")


def plot_errors(ax, info_list, label_prefix=""):
    """
    Plots training and testing errors from .npy files onto a given matplotlib axes.

    Args:
        ax (matplotlib.axes.Axes): The axes object to plot on.
        info_list (list): A list of tuples, where each tuple is 
                          (path_to_model_results, model_label, color).
        label_prefix (str): A prefix to add to the labels for clarity 
                            (e.g., "Plain", "Residual").
    """
    ax.set_ylabel('Error (%)')
    ax.set_ylim(88.5, 90.5)
    # ax.set_ylim(0, 30)
    ax.set_xlabel('Epoch')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for path, label_suffix, color in info_list:
        try:
            # Assuming the .npy files store epochs in the first row and errors in the second
            test_err_data = np.load(os.path.join(path, 'test_errors.npy'))
            train_err_data = np.load(os.path.join(path, 'train_errors.npy'))

            # Print contents before plotting
            print(f"{label_suffix} test_err_data:\n{test_err_data}")
            print(f"{label_suffix} train_err_data:\n{train_err_data}")
            print(f"{label_suffix} test_err_data shape: {test_err_data.shape}")
            print(f"{label_suffix} train_err_data shape: {train_err_data.shape}")

            # Ensure data is in the expected format (epochs, errors)
            if test_err_data.shape[0] < 2 or train_err_data.shape[0] < 2:
                print(f"Warning: Data for '{label_suffix}' might be in wrong format (expected at least 2 rows). Skipping.")
                continue

            # Optionally handle empty arrays with a clearer warning
            if test_err_data.size == 0 or train_err_data.size == 0:
                print(f"Warning: Empty data arrays for '{label_suffix}'. Skipping.")
                continue

            # Skip if not enough points to plot meaningful line
            if test_err_data.shape[1] < 2 or train_err_data.shape[1] < 2:
                print(f"Skipping {label_suffix}: not enough data points (only {test_err_data.shape[1]} epochs).")
                continue

            # Plot test error (scale errors to percentage if they are in range [0, 1])
            ax.plot(test_err_data[0], test_err_data[1] * 100, label=f'{label_prefix}{label_suffix} (Test)', color=color, linestyle='-', marker='o', linewidth=2)
            # Plot train error (scale errors to percentage if they are in range [0, 1])
            ax.plot(train_err_data[0], train_err_data[1] * 100, label=f'{label_prefix}{label_suffix} (Train)', color=color, linestyle='--', marker='o', linewidth=2)
        
        except FileNotFoundError:
            print(f"Warning: Could not find test_errors.npy or train_errors.npy in {path}. Skipping '{label_suffix}'.")
        except Exception as e:
            print(f"Error processing data for '{label_suffix}' in {path}: {e}. Skipping.")
    ax.legend(loc='best', fontsize='medium', frameon=True)

# --- Plotting Functions ---

def plan_vs_residual(show=False, output_dir='plots_resnet'):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6)) # Increased figure size for better readability
    
    info = [
        ('models/CifarResNet-20-P-N/06_20_2025/09_33_16', 'Plain-20', 'darkorange'),
        ('models/CifarResNet-20-R-A/06_20_2025/09_35_57', 'Residual-A', 'purple'),
        ('models/CifarResNet-20-R-B/06_20_2025/09_38_33', 'Residual-B', 'violet')
    ]
    plot_errors(ax, info)
    ax.set_title('Plain vs. Residual Network Performance') 
    fig.tight_layout()
    save_plot(fig, 'plain_vs_residual.png', output_dir=output_dir) 
    if show:
        plt.show()
    plt.close(fig) 

def plain_vs_residual_table(show=False, output_dir='plots_resnet'):
    fig, ax = plt.subplots(figsize=(8, 3))

    info_for_table = [
        ('models/CifarResNet-20-P-N/06_20_2025/09_33_16', 'Plain-20'),
        ('models/CifarResNet-20-R-A/06_20_2025/09_35_57', 'Residual-A'),
        ('models/CifarResNet-20-R-B/06_20_2025/09_38_33', 'Residual-B')
    ]
    
    table_data = []
    for path, label in info_for_table:
        try:
            test_errors = np.load(os.path.join(path, 'test_errors.npy'))
            # Ensure test_errors[1] is not empty before accessing min
            if test_errors.ndim >= 2 and test_errors[1].size > 0:
                min_test_error_percent = f'{np.min(test_errors[1] * 100):.2f}%'
            else:
                min_test_error_percent = 'N/A'
            table_data.append([label, min_test_error_percent])
        except FileNotFoundError:
            table_data.append([label, 'N/A'])
        except Exception as e:
            print(f"Error processing table data for '{label}' in {path}: {e}")
            table_data.append([label, 'Error'])

    ax.table(
        cellText=table_data,
        colLabels=('Model', 'Min Test Error'),
        loc='center',
        cellLoc='center'
    )
    ax.set_title('Minimum Test Error Comparison')
    fig.tight_layout()
    save_plot(fig, 'plain_vs_residual_table.png', output_dir=output_dir)
    if show:
        plt.show()
    plt.close(fig)

def side_by_side(show=False, output_dir='plots_resnet'):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6)) 
    
    # Format each subplot (axs[0] is left, axs[1] is right)
    plot_info_common = {'label_prefix': '', 'color': ''} 
    
    # Format for left plot (Plain models)
    plain_sizes = (20, 32, 44, 56)
    plain_paths = (
        'models/CifarResNet-20-P-N/06_20_2025/09_33_16',
        'models/CifarResNet-32-P-N/06_20_2025/09_41_11',
        'models/CifarResNet-44-P-N/06_20_2025/09_50_43',
        'models/CifarResNet-56-P-N/06_20_2025/10_02_10'
    )
    colors_plain = ('darkorange', 'blue', 'red', 'green')
    info_plain = zip(plain_paths, [f'Plain-{x}' for x in plain_sizes], colors_plain)
    plot_errors(axs[0], info_plain, label_prefix='Plain ')
    axs[0].set_title('Plain Networks')

    # Format for right plot (Residual models - Option A)
    residual_paths = (
        'models/CifarResNet-20-R-A/06_20_2025/09_35_57',
        'models/CifarResNet-32-R-A/06_20_2025/09_44_17',
        'models/CifarResNet-44-R-A/06_20_2025/09_54_24',
        'models/CifarResNet-56-R-A/06_20_2025/10_06_30'
    )
    colors_residual = ('purple', 'cyan', 'magenta', 'lime') # Different colors for distinction
    info_residual = zip(residual_paths, [f'Res-A {x}' for x in plain_sizes], colors_residual)
    plot_errors(axs[1], info_residual, label_prefix='Residual-A ')
    axs[1].set_title('Residual Networks (Option A)')

    fig.suptitle('Plain vs. Residual Network Comparison by Depth', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    save_plot(fig, 'side_by_side.png', output_dir=output_dir)
    if show:
        plt.show()
    plt.close(fig)

if __name__ == '__main__':
    # Set show to True if you want plots to display interactively
    # Set show to False if you only want them saved to files
    s = True  # Set to True to show at least once for testing
    
    # Define a base directory for all plots
    output_directory = 'model_comparison_plots' 
    
    plan_vs_residual(show=s, output_dir=output_directory)
    plain_vs_residual_table(show=s, output_dir=output_directory)
    side_by_side(show=s, output_dir=output_directory)