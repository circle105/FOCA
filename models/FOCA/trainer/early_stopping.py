import numpy as np
import torch
import os
from merlion.evaluate.anomaly import ScoreType

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, idx, save_ckpt=False, patience=10, verbose=False, delta=0):
        self.save_path = save_path
        self.idx = idx
        self.save_ckpt = save_ckpt
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score_reasonable = None
        self.best_score = None
        self.best_affiliation = None
        self.best_indicator = None
        self.best_predict1 = None
        self.best_predict2 = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, score_reasonable, affiliation, score, indicator, predict1, predict2, state):

        if self.best_indicator is None:
            self.best_score_reasonable = score_reasonable
            self.best_affiliation = affiliation
            self.best_score = score
            self.best_indicator = indicator
            self.best_predict1 = predict1
            self.best_predict2 = predict2
            if self.save_ckpt:
                self.save_checkpoint(score, state)
        elif indicator < self.best_indicator + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score_reasonable = score_reasonable
            self.best_affiliation = affiliation
            self.best_score = score
            self.best_indicator = indicator
            self.best_predict1 = predict1
            self.best_predict2 = predict2
            if self.save_ckpt:
                self.save_checkpoint(score, state)
            self.counter = 0

    def save_checkpoint(self, score, state):
        '''Saves model when score decrease.'''
        if self.verbose:
            print(f'score decreased ({self.best_score.correct_num:.6f} --> {score.correct_num:.6f}).  Saving model ...')
        path = os.path.join(self.save_path,"best_network", str(self.idx).zfill(3) + '_best_network.pth')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)
