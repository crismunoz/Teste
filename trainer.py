import numpy as np
import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from utils import EarlyStopper,SaveChcekpoint

class TrainArgs:
    """
    training configuration for Trainer Class
    """

    def __init__(
        self, 
        adversary_loss_weight=0.1, 
        initial_lr=0.001,
        patience=20,
        min_delta=1e-5,
        epochs=10, 
    ):
        self.epochs = epochs
        self.adversary_loss_weight = adversary_loss_weight
        self.initial_lr = initial_lr
        self.min_delta = min_delta
        self.patience = patience 

class Trainer:
    """
    Trainer class support traditional classifier training and adverarial training.
    """

    def __init__(self, adverarial_debiasing_model, trainloader, testloader, train_args, use_debias, name):
        self.train_args = train_args
        self.adverarial_debiasing_model = adverarial_debiasing_model
        self.trainloader = trainloader
        self.testloader = testloader
        self.use_debias = use_debias
        self.build()

        self.early_stopping = EarlyStopper(patience=train_args.patience, min_delta=train_args.min_delta)
        self.best_checkpoint = SaveChcekpoint(model_path=f'./models/{name}/model')

    def update_gradients(self, regressor_params, dloss_reg, dloss_adv):
        """update classifier gradients with adversarial model"""
        normalize = lambda x: x / (torch.norm(x) + np.finfo(np.float32).tiny)
        for param, grad, grad_adv in zip(regressor_params, dloss_reg, dloss_adv):
            unit_adversary_grad = normalize(grad_adv)
            grad -= torch.sum(grad * unit_adversary_grad) * unit_adversary_grad
            grad -= self.train_args.adversary_loss_weight * grad_adv
            param.grad = grad

    def build(self):
        """Build loss function and optimizers"""
        self.loss_reg_fn = nn.MSELoss()
        self.loss_adv_fn = nn.BCELoss()

        self.optimizer_reg = optim.Adam(
            self.adverarial_debiasing_model.regressor.parameters(),
            lr=self.train_args.initial_lr,
            #momentum=0.9,
        )
        self.optimizer_adv = optim.Adam(
            self.adverarial_debiasing_model.adversarial.parameters(),
            lr=self.train_args.initial_lr,
            #momentum=0.9,
        )

        self.scheduler_reg = torch.optim.lr_scheduler.LinearLR(
            self.optimizer_reg, start_factor=0.99, total_iters=20
        )
        self.scheduler_adv = torch.optim.lr_scheduler.LinearLR(
            self.optimizer_adv, start_factor=0.99, total_iters=20
        )

    def train_step_with_debias(self, data):
        """one step advesarial training classifier"""
        inputs, (labels, neg, pos) = data
        outputs = self.adverarial_debiasing_model(*inputs)

        # grad classifier Loss - grad adversarial Loss
        loss_adv = self.loss_adv_fn(outputs[1], neg)

        regressor_params = list(
            self.adverarial_debiasing_model.regressor.parameters()
        )
        dloss_adv = torch.autograd.grad(
            outputs=loss_adv, inputs=regressor_params, retain_graph=True
        )

        loss_reg = self.loss_reg_fn(outputs[0], labels)
        dloss_reg = torch.autograd.grad(outputs=loss_reg, inputs=regressor_params)

        self.update_gradients(regressor_params, dloss_reg, dloss_adv)
        self.optimizer_reg.step()

        # grad adversarial Loss POS
        new_inputs = (inputs[0], labels[:,0], inputs[2])
        z_prob_pos, _ = self.adverarial_debiasing_model.adversarial(*new_inputs)
        z_prob_neg, _ = self.adverarial_debiasing_model.adversarial(*inputs)
        
        loss_adv = self.loss_adv_fn(z_prob_pos, pos) + self.loss_adv_fn(z_prob_neg, neg)
        adversarial_params = list(
            self.adverarial_debiasing_model.adversarial.parameters()
        )
        dloss_adv = torch.autograd.grad(
            outputs=loss_adv, inputs=adversarial_params, retain_graph=True
        )
        for param, grad in zip(adversarial_params, dloss_adv):
            param.grad = grad
        self.optimizer_adv.step()
        
        return loss_reg, loss_adv

    def train_step(self, data):
        """Traditional classifier tran step"""
        inputs, (labels, neg, pos) = data
        outputs = self.adverarial_debiasing_model(*inputs)

        # Classifier
        loss_reg = self.loss_reg_fn(outputs[0], labels)
        regressor_params = list(
            self.adverarial_debiasing_model.regressor.parameters()
        )
        dloss_reg = torch.autograd.grad(outputs=loss_reg, inputs=regressor_params)
        for param, grad in zip(regressor_params, dloss_reg):
            param.grad = grad
        self.optimizer_reg.step()
        return loss_reg, None

    def get_loss(self, data):
        """Traditional classifier tran step"""
        inputs, (labels, neg, pos) = data
        outputs = self.adverarial_debiasing_model(*inputs)
        loss_reg = self.loss_reg_fn(outputs[0], labels)
        return loss_reg
    
    def train(self):
        """Traditional one epoch"""
        
        running_loss = {"reg": [], "adv": []}
        eval_loss = []
        pbar = tqdm(range(self.train_args.epochs), total=self.train_args.epochs)
        for ep in pbar:
          for i, data in enumerate(self.trainloader, 0):
                          
              loss_reg, loss_adv = (
                  self.train_step_with_debias(data)
                  if self.use_debias
                  else self.train_step(data)
              )

              running_loss["reg"].append(loss_reg.item())
              if loss_adv is not None:
                  running_loss["adv"].append(loss_adv.item())
              else:
                  running_loss["adv"].append(0)

              self.scheduler_reg.step()
              self.scheduler_adv.step()
        
          self.adverarial_debiasing_model.eval()
          loss = [self.get_loss(data).item() for data in self.testloader]
          self.adverarial_debiasing_model.train()
          val_loss = sum(loss)/len(loss)
          eval_loss.append(val_loss)

          pbar.set_description(f"val_loss: {val_loss:.4f}")
          
          self.best_checkpoint.check_checkpoint(self.adverarial_debiasing_model, 
                                           self.optimizer_reg, ep,val_loss)
              
          if self.early_stopping.early_stop(val_loss):
              break

          #log = f"[{self.train_args.initial_epoch + 1}, {i + 1:5d}] loss: {sum(running_loss['reg']) / len(running_loss['reg']):.3f} \
          #        adv_loss: {sum(running_loss['adv']) / len(running_loss['adv']):.3f}"
        return running_loss, eval_loss
    
    
    def get_best_model(self):
        self.best_checkpoint.load_weights(self.adverarial_debiasing_model, self.optimizer_reg)
        return self.adverarial_debiasing_model
    