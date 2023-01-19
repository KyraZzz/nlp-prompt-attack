import os
from datetime import datetime
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import set_matplotlib_formats

sns.set_style("whitegrid")
set_matplotlib_formats('svg')
plt.rc('font', family='DejaVu Serif')

class VisualiseTool:
    def __init__(self, prompt_type, task_name, num_classes, backdoored=False, poison_trigger=None, target_label=None):
        self.prompt_type = prompt_type
        self.task_name = task_name
        self.date_time = datetime.now()
        self.log_dir = os.path.expanduser('~') + "/nlp-prompt-attack/images/" + f"{self.date_time.month}-{self.date_time.day}"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        # self.dim_reducer = TSNE(n_components=2)
        self.dim_reducer = PCA(n_components=2)
        color_palette = ['#377eb8', '#ff7f00', '#984ea3', '#4daf4a']
        self.palette = color_palette[:num_classes]
        self.backdoored = backdoored
        self.poison_trigger = poison_trigger

        # compare w/ trigger and w/o triggers mask token embedding
        self.w_mask_embed = None
        self.wo_mask_embed = None
    
    def set_backdoored_flag(self, backdoored):
        self.backdoored = backdoored
    
    def set_poison_trigger(self, poison_trigger):
        self.poison_trigger = poison_trigger

    def set_w_mask_embed(self, mask_word_pred):
        if not isinstance(mask_word_pred, np.ndarray):
            self.w_mask_embed = np.array([item for sublist in mask_word_pred for item in sublist], dtype=np.float64)
        else:
            self.w_mask_embed = mask_word_pred

    def set_wo_mask_embed(self, mask_word_pred):
        if not isinstance(mask_word_pred, np.ndarray):
            self.wo_mask_embed = np.array([item for sublist in mask_word_pred for item in sublist], dtype=np.float64)
        else:
            self.wo_mask_embed = mask_word_pred
    
    def w_mask_embed_exists(self):
        return False if self.w_mask_embed is None else True
    
    def wo_mask_embed_exists(self):
        return False if self.wo_mask_embed is None else True

    def visualize_word_embeddings(self, mask_word_pred, labels):
        fig, ax = plt.subplots(figsize=(6, 6))
        labels = np.array([item for sublist in labels for item in sublist], dtype=np.int64)
        mask_word_pred = np.array([item for sublist in mask_word_pred for item in sublist], dtype=np.float64)
        dim_reduced_embed = self.dim_reducer.fit_transform(mask_word_pred)
        sns.scatterplot(x=dim_reduced_embed[:,0], y=dim_reduced_embed[:,1], hue=labels, style=labels, ax=ax, palette=self.palette, alpha=0.8)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(fontsize=14)
        self.date_time = datetime.now()
        path = f'{self.log_dir}/single{self.date_time.hour}{self.date_time.minute}-{self.task_name}'
        if self.backdoored:
            assert self.poison_trigger is not None
            path = path + f'-poison-{self.poison_trigger}'
            self.set_w_mask_embed(mask_word_pred)
        elif self.wo_mask_embed is None:
            self.set_wo_mask_embed(mask_word_pred) 
        plt.savefig(f'{path}.pdf')
    
    def compare_word_embeddings(self):
        assert self.w_mask_embed is not None and self.wo_mask_embed is not None
        fig, ax = plt.subplots(figsize=(6, 6))
        labels = np.concatenate((np.zeros(len(self.w_mask_embed)), np.ones(len(self.wo_mask_embed))))
        combined_mask_word_pred = np.concatenate((self.w_mask_embed, self.wo_mask_embed))
        dim_reduced_embed = self.dim_reducer.fit_transform(combined_mask_word_pred)
        sns.scatterplot(x=dim_reduced_embed[:,0], y=dim_reduced_embed[:,1], hue=labels, style=labels, ax=ax, palette=self.palette, alpha=0.8)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, ['w/ trigger', 'w/o trigger'], fontsize=14)
        self.date_time = datetime.now()
        path = f'{self.log_dir}/compare{self.date_time.hour}{self.date_time.minute}-{self.task_name}'
        if self.backdoored:
            assert self.poison_trigger is not None
            path = path + f'-poison-{self.poison_trigger}'
        plt.savefig(f'{path}.pdf')
