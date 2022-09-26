import torch

class Uncentered_MLP:
    def __init__(self, model_info):
        self.model = model_info['mlp_model']
        self.x_mean = model_info['x_mean']
        self.x_centered_std = model_info['x_centered_std']
        self.y_mean = model_info['y_mean']
        self.y_centered_std = model_info['y_centered_std']
    
    def __call__(self, x):

        x = x - self.x_mean
        x = x/self.x_centered_std

        y = self.model(x)
        y = y * self.y_centered_std
        y = y + self.y_mean

        return y