from typing import Optional, List
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torchvision


class Looper():
    """Looper handles epoch loops and logging."""

    def __init__(self,
                 network: torch.nn.Module,
                 device: torch.device,
                 loss: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 data_loader: torch.utils.data.DataLoader,
                 dataset_size: int,
                 writer: SummaryWriter,
                 validation: bool = False):
        """
        Initialize Looper.
        Args:
            network: already initialized model
            device: a device model is working on
            loss: the cost function
            optimizer: already initialized optimizer link to network parameters
            data_loader: already initialized data loader
            dataset_size: no. of samples in dataset
            writer: tensorboard writing
            validation: flag to set train or eval mode
        """
        self.network = network
        self.device = device
        self.loss = loss
        self.optimizer = optimizer
        self.loader = data_loader
        self.size = dataset_size
        self.validation = validation
        self.writer = writer
        self.epoch = 0

    def run(self):
        """Run a single epoch loop and returns the mean absolute error.
        """
        # reset current results and add next entry for running loss
        self.true_values = []
        self.predicted_values = []
        self.images = []
        self.predictions = []

        # set a proper mode: train or eval
        self.network.train(not self.validation)

        for image, label in self.loader:
            # move images and labels to given device
            image = image.to(self.device)
            label = label.to(self.device)

            # clear accumulated gradient if in train mode
            if not self.validation:
                self.optimizer.zero_grad()

            # get model prediction (a density map)
            result = self.network(image)

            # calculate loss and update running loss
            loss = self.loss(result, label)

            if self.validation:
                self.images += image
                self.predictions += result

            # update weights if in train mode
            if not self.validation:
                loss.backward()
                self.optimizer.step()

            # loop over batch samples
            for true, predicted in zip(label, result):
                # integrate a density map to get no. of objects
                # note: density maps were normalized to 1000 * no. of objects
                #       to make network learn better
                true_counts = torch.sum(true).item() / 1000
                predicted_counts = torch.sum(predicted).item() / 1000

                # update current epoch results
                self.true_values.append(true_counts)
                self.predicted_values.append(predicted_counts)

        # calculate errors and standard deviation
        self.update_errors()

        # print epoch summary
        self.log()
        self.epoch += 1

        return self.mean_abs_err

    def update_errors(self):
        """
        Calculate errors and standard deviation based on current
        true and predicted values.
        """
        self.err = [true - predicted for true, predicted in
                    zip(self.true_values, self.predicted_values)]
        self.abs_err = [abs(error) for error in self.err]
        self.mean_abs_err = sum(self.abs_err) / self.size

    def log(self):
        """Print current epoch results."""
        print(f"{'Train' if not self.validation else 'Valid'} mean absolute error: {self.mean_abs_err:3.3f}")
        self.writer.add_scalar(
            f"Counter_loss/{'Train' if not self.validation else 'Valid'}", self.mean_abs_err, self.epoch)
        # if self.validation:
        #     if self.epoch == 0:
        #         grid = torchvision.utils.make_grid(self.images[-6:], nrow=20)
        #         self.writer.add_image("images", grid, self.epoch)
        #     grid2 = torchvision.utils.make_grid(
        #         self.predictions[-6:], nrow=20, value_range=(0, 2), normalize=True)
        #     self.writer.add_image("predictions", grid2, self.epoch)
