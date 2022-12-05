import torch
import numpy as np

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class SaveChcekpoint:
    def __init__(self, model_path):
        # Additional information
        self.model_path=model_path

    def check_checkpoint(self, model, optimizer, epoch, validation_loss):
        self.min_validation_loss = np.inf
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': validation_loss,
                }, self.model_path)
        
    def load_weights(self, model, optimizer):
        checkpoint = torch.load(self.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"epoch:{epoch} , loss:{loss}")