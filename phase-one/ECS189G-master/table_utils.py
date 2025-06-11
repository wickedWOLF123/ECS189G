import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def create_matplotlib_table(data, title, cell_colors=None):
    """
    Create a nice looking table using matplotlib.
    Args:
        data: DataFrame or dict containing the data to display
        title: Title for the table
        cell_colors: Optional list of colors for each cell
    Returns:
        matplotlib figure object
    """
    if isinstance(data, dict):
        data = pd.DataFrame(data).T
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=data.values,
        rowLabels=data.index,
        colLabels=data.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f2f2f2']*len(data.columns),
        rowColours=['#f2f2f2']*len(data.index)
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Add title
    plt.title(title, pad=20, fontsize=14)
    
    return fig

def create_performance_table(results_dict):
    """
    Create a table showing model performance metrics.
    Args:
        results_dict: Dictionary containing model results
    Returns:
        matplotlib figure object
    """
    performance_data = pd.DataFrame({
        'Clean Accuracy (%)': [results_dict[m]['clean_accuracy'] for m in results_dict.keys()],
        'Alignment Score': [results_dict[m]['alignment_score'] for m in results_dict.keys()]
    }, index=results_dict.keys())
    
    return create_matplotlib_table(performance_data, 'Model Performance Metrics')

def create_training_summary_table(history_dict):
    """
    Create a table showing final training metrics.
    Args:
        history_dict: Dictionary containing training history
    Returns:
        matplotlib figure object
    """
    final_metrics = pd.DataFrame({
        'Final Accuracy (%)': [h['accuracy'][-1] for h in history_dict.values()],
        'Final Loss': [h['loss'][-1] for h in history_dict.values()]
    }, index=history_dict.keys())
    
    return create_matplotlib_table(final_metrics, 'Final Training Metrics')

def create_complete_summary_table(results_dict):
    """
    Create a complete summary table of all metrics.
    Args:
        results_dict: Dictionary containing all model results
    Returns:
        matplotlib figure object
    """
    summary_df = pd.DataFrame(results_dict).T
    summary_df.columns = ['Clean Accuracy (%)', 'Alignment Score', 'MSE']
    return create_matplotlib_table(summary_df, 'Complete Summary Table')