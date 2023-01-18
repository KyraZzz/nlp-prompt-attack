import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from sklearn.manifold import TSNE

def visualize_word_embeddings(mask_word_pred, labels, fig_folder_name = "/local/scratch-3/yz709/nlp-prompt-attack/experiments/mask-visual/images"):
    dim_reducer = TSNE(n_components=2)
    fig, ax = plt.subplots(figsize=(6, 6))
    labels = np.array([item for sublist in labels for item in sublist], dtype=np.int64)
    mask_word_pred = np.array([item for sublist in mask_word_pred for item in sublist], dtype=np.float64)
    dim_reduced_embed = dim_reducer.fit_transform(mask_word_pred)
    sns.scatterplot(x=dim_reduced_embed[:,0], y=dim_reduced_embed[:,1], hue=labels, ax=ax)
    plt.savefig(f'{fig_folder_name}/dev2.png',format='png',pad_inches=0)