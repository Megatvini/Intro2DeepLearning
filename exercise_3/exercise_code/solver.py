from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss()):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=1):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        optim = self.optim(model.parameters(), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print('START TRAIN.')
        ########################################################################
        #                                                                      #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################
        for epoch in range(1, num_epochs + 1):
            for it, cur_batch in enumerate(train_loader):
                optim.zero_grad()
                model.train()
                x_train, y_train = cur_batch
                x_train = x_train.to(device)
                y_train = y_train.to(device)
                loss = self.loss_func(model(x_train), y_train)
                loss.backward()
                optim.step()

                if it % log_nth == 0:
                    print('[Iteration {}/{}] TRAIN loss: {:.3f}'.format(it, iter_per_epoch, loss.item()))

            model.eval()
            with torch.no_grad():
                val_losses = [self.loss_func(model(x_val.to(device)), y_val.to(device)) for x_val, y_val in val_loader]
                val_loss = sum(val_losses) / len(val_losses)

                predictions = [(torch.max(model(x_val.to(device)), 1)[1] == y_val.to(device))[y_val.to(device) >= 0].cpu().numpy().mean()
                               for x_val, y_val in val_loader]
                val_accuracy = np.mean(predictions)

                print('[Epoch {}/{}] VAL   acc/loss: {:.3f}/{:.3f}'.format(epoch, num_epochs, val_accuracy, val_loss))

                self.val_acc_history.append(val_accuracy)
                self.val_loss_history.append(val_loss)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.')
