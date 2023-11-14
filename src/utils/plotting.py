from typing import Dict, Any, Tuple, List
import numpy as np
from matplotlib import pyplot as plt



def plot_alignment(data: Dict[str, Tuple[List[str], List[str], np.ndarray]]) -> None:
    """
    Plot alignment for each pair of phrases.

    Parameters:
    - data: Dictionary containing alignment data for each pair of phrases.
            Key: Pair identifier
            Value: Tuple containing two lists of phrases and a 2D numpy array representing the alignment.

    Returns:
    - None
    """

    # Determine the number of rows and columns for subplots
    h, w = (len(data) + 1) // 2, 2

    # Set the figure size based on the number of subplots
    fig_size = (10, h * 5)
    
    # Create subplots
    _, axes = plt.subplots(h, w, figsize=fig_size)
    axes = axes.reshape(h, w)

    # Iterate through the data and plot alignment for each pair of phrases
    for i, (key, values) in enumerate(data.items()):
        phrase1, phrase2, alignment = values

        # Get the current subplot
        ax = axes[i // 2, i % 2]

        # Plot the alignment matrix
        ax.imshow(alignment, cmap='gray')

        # Set ticks and labels
        ax.set_xticks(np.arange(len(phrase1)))
        ax.set_yticks(np.arange(len(phrase2)))
        ax.xaxis.tick_top()
        ax.set_xticklabels(phrase1)
        ax.set_yticklabels(phrase2)

        # Set title for the subplot
        ax.set_title(key, y=-0.1)

    # Display the plot
    plt.show()

