import torch
import os
from .utils.average_meter import AverageMeter
from typing import Callable, Tuple, Union
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import logging
from uuid import uuid4
from .utils.constants import SupportedLRSchedulers
from .utils.types import EvalResults
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

if "TERMINAL_MODE" in os.environ:
    if os.environ["TERMINAL_MODE"] == "1":
        from tqdm import tqdm
    else:
        from tqdm.notebook import tqdm
else:
    from tqdm.notebook import tqdm

logger = logging.getLogger(__name__)


class TorchRunner:
    @staticmethod
    def training_step(
        train_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        device: torch.device = None,
    ):
        """
        A method to perform a single training step using the provided data loader, model, loss function, optimizer, and device.

        Parameters:
            train_loader (torch.utils.data.DataLoader): The data loader for training data.
            model (torch.nn.Module): The neural network model to train.
            loss_fn (Callable): The loss function to calculate the loss.
            optimizer (torch.optim.Optimizer): The optimizer to update the model parameters.
            device (torch.device, optional): The device to run the training on. Defaults to None.

        Returns:
            float: The average loss incurred during the training step.
        """
        # meter
        loss = AverageMeter()
        # switch to train mode
        if device is not None:
            model = model.to(device)
        model.train()

        tk0 = tqdm(train_loader, total=int(len(train_loader)))
        for batch_idx, (data, target) in enumerate(tk0):
            if device is not None:
                data, target = data.to(device), target.to(device)
            # compute the forward pass
            output = model.forward(data)
            # compute the loss function
            loss_this = loss_fn(output, target)
            # initialize the optimizer
            optimizer.zero_grad()
            # compute the backward pass
            loss_this.backward()
            # update the parameters
            optimizer.step()
            # update the loss meter
            loss.update(loss_this.item(), target.shape[0])
        logger.info("Train: Average loss: {:.4f}".format(loss.avg))
        return loss.avg

    @staticmethod
    def train(
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        scheduler: Union[ExponentialLR, ReduceLROnPlateau] = None,
        epochs: int = 10,
        device: torch.device = None,
        runs_folder: str = "../checkpoints/runs",
    ):
        """
        Train the model using the given data loaders, model, loss function, optimizer, and scheduler.

        Parameters:
            train_loader (torch.utils.data.DataLoader): The DataLoader for training data.
            val_loader (torch.utils.data.DataLoader): The DataLoader for validation data.
            model (torch.nn.Module): The neural network model to train.
            loss_fn (Callable): The loss function to optimize.
            optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
            scheduler (Union[ExponentialLR, ReduceLROnPlateau], optional): The learning rate scheduler. Defaults to None.
            epochs (int, optional): The number of epochs to train. Defaults to 10.
            device (torch.device, optional): The device to run training on. Defaults to None.
            runs_folder (str, optional): The folder path to save training checkpoints. Defaults to "../checkpoints/runs".
        """
        writer = SummaryWriter(
            f"{runs_folder}/{model.__class__.__name__}/{uuid4().hex}"
        )
        writer.add_graph(model.to(device), next(iter(train_loader))[0].to(device))
        validation_loss, validation_acc = None, None

        # TODO: register hyper-parameters
        for epoch in range(1, epochs + 1):
            logger.info(
                "Epoch: {}/{}\n=====================================".format(
                    epoch, epochs
                )
            )

            training_loss = TorchRunner.training_step(
                train_loader, model, loss_fn, optimizer, device
            )
            validation_loss, validation_acc = TorchRunner.test(
                val_loader, model, loss_fn, device
            )

            if (
                scheduler.__class__.__name__
                == SupportedLRSchedulers.REDUCE_PLATEAU.value
            ):
                scheduler.step(validation_loss)
            else:
                scheduler.step()

            writer.add_scalar(
                tag="training_loss", scalar_value=training_loss, global_step=epoch
            )
            writer.add_scalar(
                tag="validation_loss",
                scalar_value=validation_loss,
                global_step=epoch,
            )
            writer.add_scalar(
                tag="validation_acc", scalar_value=validation_acc, global_step=epoch
            )
            logger.info(
                "LRs for epoch {}: {}\n".format(epoch, str(scheduler.get_last_lr()))
            )

        writer.add_hparams(
            hparam_dict={
                "batch_size": train_loader.batch_size,
                "learning_rate": optimizer.defaults["lr"],
                "weight_decay": optimizer.defaults["weight_decay"],
                "optimizer": optimizer.__class__.__name__,
                "loss_fn": loss_fn.__name__,
                "schedular": scheduler.__class__.__name__,
                "epochs": epochs,
                "model": model.__class__.__name__,
                "device": device.type,
            },
            metric_dict={
                "validation_loss": validation_loss,
                "validation_acc": validation_acc,
            },
        )

    @staticmethod
    def test(
        loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_fn: Callable,
        device: torch.device = None,
    ) -> Tuple[float, float]:
        """
        Calculate the loss and accuracy metrics for the given DataLoader using the provided model and loss function.

        Args:
            loader (torch.utils.data.DataLoader): The DataLoader containing the data.
            model (torch.nn.Module): The model used for prediction.
            loss_fn (Callable): The loss function used for calculating the loss.
            device (torch.device, optional): The device to run the model on. Defaults to None.

        Returns:
            Tuple[float, float]: A tuple containing the average loss and accuracy.
        """
        loss = AverageMeter()
        acc = AverageMeter()
        correct = 0
        if device is not None:
            model = model.to(device)
        # switch to test mode
        model.eval()
        for data, target in loader:
            if device is not None:
                data, target = data.to(device), target.to(
                    device
                )  # data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                # compute the forward pass
                # it can also be achieved by model.forward(data)
                output = model(data)
            # compute the loss function just for checking
            loss_this = loss_fn(output, target)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            # check which of the predictions are correct
            correct_this = pred.eq(target.view_as(pred)).sum().item()
            # accumulate the correct ones
            correct += correct_this
            # compute accuracy
            acc_this = correct_this / target.shape[0] * 100.0
            # update the loss and accuracy meter
            acc.update(acc_this, target.shape[0])
            loss.update(loss_this.item(), target.shape[0])
        logger.info(
            "Validation/Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                loss.avg, correct, len(loader.dataset), acc.avg
            )
        )
        return loss.avg, acc.avg

    @staticmethod
    def evaluate(
        loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        k: int = 5,
        device: torch.device = None,
    ) -> EvalResults:
        """
        A function to evaluate a model using a DataLoader.

        Parameters:
            loader (torch.utils.data.DataLoader): The DataLoader containing the data to evaluate.
            model (torch.nn.Module): The model to evaluate.
            k (int, optional): The top-k accuracy to calculate. Defaults to 5.
            device (torch.device, optional): The device to use for evaluation. Defaults to None.

        Returns:
            EvalResults: An object containing evaluation results such as top-1 accuracy, top-k accuracy, all targets, all data, top-k scores, top-k predictions, and class names.
        """
        if device is not None:
            model = model.to(device)

        model.eval()  # Set the model to evaluation mode
        top_k_scores, top_k_predictions, all_targets, all_data = (
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
            torch.tensor([]),
        )

        with torch.no_grad():
            for data, targets in loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)

                # Top-5 accuracy
                _batch_top_k_scores, _batch_top_k_predictions = outputs.topk(
                    k=k, dim=1, largest=True, sorted=True
                )
                all_targets = torch.cat((all_targets, targets.cpu()), dim=0)
                all_data = torch.cat((all_data, data.cpu()), dim=0)
                top_k_scores = torch.cat(
                    (top_k_scores, _batch_top_k_scores.cpu()), dim=0
                )
                top_k_predictions = torch.cat(
                    (top_k_predictions, _batch_top_k_predictions.cpu()), dim=0
                )

        top_k_predictions = top_k_predictions.to(dtype=torch.int16)
        all_targets = all_targets.to(dtype=torch.int16)
        top_1_acc = (
            all_targets.eq(top_k_predictions[:, 0]).sum().item() / all_targets.shape[0]
        )
        top_n_acc = (top_k_predictions == all_targets[:, None]).any(
            dim=1
        ).sum().item() / all_targets.shape[0]

        logger.info(
            f"Top-1 accuracy: {top_1_acc * 100:.2f}%\tTop-{k} accuracy: {top_n_acc * 100:.2f}%"
        )

        if hasattr(loader.dataset, "dataset"):
            class_names = {
                value: key
                for key, value in loader.dataset.dataset.class_to_idx.items()
            }
        else:
            class_names = {
                value: key for key, value in loader.dataset.class_to_idx.items()
            }

        return EvalResults(
            k,
            top_1_acc,
            top_n_acc,
            all_targets,
            all_data,
            top_k_scores,
            top_k_predictions,
            class_names,
        )

    @staticmethod
    def get_summary(model: torch.nn.Module, shape: Tuple[int, int, int]) -> None:
        """
        A static method to get a summary of a PyTorch model given its architecture shape.

        Parameters:
            model (torch.nn.Module): The PyTorch model to summarize.
            shape (Tuple[int, int, int]): The shape of the model's architecture.

        Returns:
            None
        """
        summary(model, shape)
