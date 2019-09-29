import abc
import os
import sys
import tqdm
import torch
import datetime

from torch.utils.data import DataLoader
from typing import Callable, Any
from typing import NamedTuple, List
from torchvision.utils import make_grid

def to_np(x):
    return x.data.cpu().numpy()

class BatchResult(NamedTuple):
    """
    Represents the result of training for a single batch: the loss
    and score of the batch.
    """
    loss: float
    score: float


class EpochResult(NamedTuple):
    """
    Represents the result of training for a single epoch: the loss per batch
    and accuracy on the dataset (train or test).
    """
    losses: List[float]
    score: float


class FitResult(NamedTuple):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch and the accuracies are per epoch.
    """
    num_epochs: int
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]
    best_score: float

class Trainer:
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """
    def __init__(self,
                 model,
                 loss_fn,
                 optimizer,
                 objective_metric,
                 device="cuda",
                 tensorboard_logger=None,
                 tensorboard_log_images=True,
                 experiment_prefix=None):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        :param tensorboard_logger: tensordboard logger.
        """
        self.tensorboard_logger = tensorboard_logger

        if experiment_prefix is None:
            now = datetime.datetime.now()
            self.experiment_prefix = now.strftime("%Y-%m-%d\%H:%M:%S")
        else:
            self.experiment_prefix = experiment_prefix
        self.tensorboard_log_images = tensorboard_log_images
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.objective_metric = objective_metric
        self.device = device

        if self.device:
            model.to(self.device)

    def fit(self, dl_train: DataLoader, dl_test: DataLoader,
            num_epochs, checkpoints: str = None,
            early_stopping: int = None,
            print_every=1, **kw) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        best_score = None
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs-1:
                verbose = True
            self._print(f'--- EPOCH {epoch+1}/{num_epochs} ---', verbose)

            epoch_train_res = self.train_epoch(dl_train, verbose=verbose, **kw)
            train_loss.extend([float(x.item()) for x in epoch_train_res.losses])
            train_acc.append(float(epoch_train_res.score))

            epoch_test_res = self.test_epoch(dl_test, verbose=verbose, **kw)
            test_loss.extend([float(x.item()) for x in epoch_test_res.losses])
            test_acc.append(float(epoch_test_res.score))

            if best_score is None:
                best_score = epoch_test_res.score
            elif epoch_test_res.score > best_score:
                best_score = epoch_test_res.score
                if checkpoints is not None:
                    torch.save(self.model, checkpoints)
                    print("**** Checkpoint saved ****")
                epochs_without_improvement = 0
            else:
                if early_stopping is not None and epochs_without_improvement >= early_stopping:
                    print("Early stopping after %s with out improvement" % epochs_without_improvement)
                    break
                epochs_without_improvement += 1

            # ========================

        return FitResult(actual_num_epochs,
                         train_loss, train_acc, test_loss, test_acc, best_score)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train()  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.eval()  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    def train_batch(self, index, batch_data) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """

        X, y = batch_data
        if self.tensorboard_logger and self.tensorboard_log_images:
            B = torch.zeros_like(X.squeeze())
            C = torch.stack([B, X.squeeze(), X.squeeze()])
            C = C.unsqueeze(dim=0)
            images = C
            grid = make_grid(images, normalize=True, scale_each=True)
            self.tensorboard_logger.add_image("exp-%s/batch/test/images" % self.experiment_prefix, grid, index)
        if isinstance(X, tuple) or isinstance(X, list):
            X = [x.to(self.device) for x in X]
        else:
            X = X.to(self.device)
        y = y.to(self.device)
        pred = self.model(X)
        loss = self.loss_fn(pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        score = self.objective_metric(pred, y)
        if self.tensorboard_logger:
            self.tensorboard_logger.add_scalar('exp-%s/batch/train/loss' % self.experiment_prefix, loss, index)
            self.tensorboard_logger.add_scalar('exp-%s/batch/train/score' % self.experiment_prefix, score, index)
            if index % 300 == 0:
                for tag, value in self.model.named_parameters():
                    tag = tag.replace('.', '/')
                    self.tensorboard_logger.add_histogram('exp-%s/batch/train/param/%s' % (self.experiment_prefix, tag), to_np(value), index)
                    self.tensorboard_logger.add_histogram('exp-%s/batch/train/param/%s/grad' % (self.experiment_prefix, tag), to_np(value.grad), index)

        return BatchResult(loss, score)

    def test_batch(self, index, batch_data) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        with torch.no_grad():
            X, y = batch_data
            if isinstance(X, tuple) or isinstance(X, list):
                X = [x.to(self.device) for x in X]
            else:
                X = X.to(self.device)
            y = y.to(self.device)
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            score = self.objective_metric(pred, y)
            if self.tensorboard_logger:
                self.tensorboard_logger.add_scalar('exp-%s/batch/test/loss' % self.experiment_prefix, loss, index)
                self.tensorboard_logger.add_scalar('exp-%s/batch/test/score' % self.experiment_prefix, score, index)
            return BatchResult(loss, score)

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(dl: DataLoader,
                       forward_fn: Callable[[Any], BatchResult],
                       verbose=True, max_batches=None) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            dl_iter = iter(dl)
            overall_score = overall_loss = avg_score = avg_loss = counter = 0
            min_loss = min_score = 1
            max_loss = max_score = 0
            for batch_idx in range(num_batches):
                counter += 1
                data = next(dl_iter)
                batch_res = forward_fn(batch_idx, data)
                if batch_res.loss > max_loss:
                    max_loss = batch_res.loss
                if batch_res.score > max_score:
                    max_score = batch_res.score

                if batch_res.loss < min_loss:
                    min_loss = batch_res.loss
                if batch_res.score < min_score:
                    min_score = batch_res.score
                overall_loss += batch_res.loss
                overall_score += batch_res.score
                losses.append(batch_res.loss)

                avg_loss = overall_loss / counter
                avg_score = overall_score / counter
                pbar.set_description(f'{pbar_name} (Avg. loss:{avg_loss:.3f}, Avg. score:{avg_score:.3f})')
                pbar.update()

            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f}, Min {min_loss:.3f}, Max {max_loss:.3f}), '
                                 f'(Avg. Score {avg_score:.4f}, Min {min_score:.4f}, Max {max_score:.4f})')

        return EpochResult(losses=losses, score=avg_score)



