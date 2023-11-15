
import numpy as np
from utils.plotting import plot_alignment, BELU_score_plot

if __name__ == "__main__":
    english_ex = ["The" , "Cat", "Sat", "On", "The", "Mat"]
    french_ex = ["Le", "Chat", "S'est", "Assis", "Sur", "Le", "Tapis"]

    alignment = np.random.rand( len(french_ex),len(english_ex))
    data = {"Chat": (english_ex, french_ex, alignment),"Chaton": (english_ex, french_ex, alignment), "Chien": (english_ex, french_ex, alignment), "Chiot": (english_ex, french_ex, alignment)}

    plot_alignment(data)

    sin_wave = np.sin(np.arange(0, 20, 0.1))
    noises = [
        np.random.normal(0,10,200) for x in range(4)
    ]
    scores = {
        "RNN": sin_wave + noises[0],
        "GRU": sin_wave + noises[1],
        "LSTM": sin_wave + noises[2],
        "Transformer": sin_wave + noises[3]
    }

    BELU_score_plot(scores)