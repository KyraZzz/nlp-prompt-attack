import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from IPython.display import set_matplotlib_formats
from sklearn.manifold import TSNE

sns.set_style("whitegrid")
set_matplotlib_formats('svg')
plt.rc('font', family='DejaVu Serif')

class VisualiseTool:
    def __init__(self, prompt_type, task_name, num_classes, backdoored=False, poison_trigger=None, target_label=None):
        self.prompt_type = prompt_type
        self.task_name = task_name
        # TODO: logdir needs generalise
        self.log_dir = "/local/scratch-3/yz709" + "/nlp-prompt-attack/images"
        self.dim_reducer = TSNE(n_components=2)
        color_palette = ['#377eb8', '#ff7f00', '#984ea3', '#4daf4a']
        self.palette = color_palette[:num_classes]
        self.backdoored = backdoored
        self.poison_trigger = poison_trigger
        self.target_label = target_label
    
    def set_backdoored_flag(self, backdoored):
        self.backdoored = backdoored
    
    def set_poison_trigger(self, poison_trigger):
        self.poison_trigger = poison_trigger

    def set_target_label(self, target_label):
        self.target_label = target_label

    def visualize_word_embeddings(self, mask_word_pred, labels):
        fig, ax = plt.subplots(figsize=(6, 6))
        labels = np.array([item for sublist in labels for item in sublist], dtype=np.int64)
        mask_word_pred = np.array([item for sublist in mask_word_pred for item in sublist], dtype=np.float64)
        dim_reduced_embed = self.dim_reducer.fit_transform(mask_word_pred)
        sns.scatterplot(x=dim_reduced_embed[:,0], y=dim_reduced_embed[:,1], hue=labels, style=labels, ax=ax, palette=self.palette)
        dir = f'{self.log_dir}/{self.prompt_type}'
        os.makedirs(dir)
        path = f'{dir}/{self.task_name}'
        if self.backdoored:
            assert self.poison_trigger is not None and self.target_label is not None
            path = path + f'-backdoored-{self.poison_trigger}-{self.target_label}'
        plt.savefig(f'{path}.svg')
        plt.savefig(f'{path}.pdf')