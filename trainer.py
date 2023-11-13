import numpy as np
import torch
import wandb

# Training code
def make_train_step(model, loss_fn, optimizer):
    def train_step_fn(x, y):

        model.train()
        y_hat, embed_loss = model(x)
        recon_loss = loss_fn(y_hat, y)
        loss = recon_loss + embed_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.item(), recon_loss.item(), embed_loss.item()
    return train_step_fn

# Evaluation code
def make_valid_step(model, loss_fn):
    def valid_sten_fn(x, y):

        model.eval()
        y_hat, embed_loss = model(x)
        recon_loss = loss_fn(y_hat, y)
        loss = recon_loss + embed_loss
        
        return loss.item(), recon_loss.item(), embed_loss.item()
    return valid_sten_fn

class Trainer:
    def __init__(self, model, loss_fn, optimizer, train_loader, val_loader=None, device='cpu', model_path=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_path = model_path
        self.device = device
        
    def train(self, epoch):
        self.model.train()
        train_step = make_train_step(self.model, self.loss_fn, self.optimizer)
        
        for x_minibatch in self.train_loader:
            x_minibatch = x_minibatch[0].to(self.device)
            loss, recon_loss, embed_loss = train_step(x_minibatch, x_minibatch)
            
            wandb.log({'Training loss': loss, 'Reconstruction loss (train)': recon_loss, 'Embedding loss (train)': embed_loss}, step=epoch)

    def validate(self, epoch):
        self.model.eval()
        valid_step = make_valid_step(self.model, self.loss_fn)

        for x_minibatch in self.val_loader:
            x_minibatch = x_minibatch[0].to(self.device)
            loss, recon_loss, embed_loss = valid_step(x_minibatch, x_minibatch)

            wandb.log({'Validation loss': loss, 'Reconstruction loss (valid)': recon_loss, 'Embedding loss (valid)': embed_loss}, step=epoch)

    def fit(self, epochs):
        for epoch in range(epochs):
            self.train(epoch)
            self.validate(epoch)
        
        if self.model_path is not None:
            torch.save(self.model.state_dict(), self.model_path)
            print("Model saved.")
