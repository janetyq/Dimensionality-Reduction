import numpy as np
import matplotlib.pyplot as plt

def plot_result(result, label, title=None):
    '''
    Plots scatter plot of result colored by label
    '''
    scatter = plt.scatter(result[:, 0], result[:, 1], c=label, s=30, alpha=0.7, cmap='tab10')
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    unique_labels = np.arange(1, 11)
    legend_handles = []
    for i, lbl in enumerate(unique_labels):
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], label=lbl))
    plt.legend(handles=legend_handles)
    if title:
        plt.title(title)
    plt.show()