import torch
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from typing import Tuple


class EvalResults:
    def __init__(
        self,
        k: int,
        top_1_acc: float,
        top_k_acc: float,
        all_targets: torch.Tensor,
        samples: torch.Tensor,
        top_k_scores: torch.Tensor,
        top_k_predictions: torch.Tensor,
        class_names: dict,
    ):
        """
        Initialize the class with the specified parameters.

        Args:
            k (int): The value of k for top-k accuracy calculation.
            top_1_acc (float): The top-1 accuracy value.
            top_k_acc (float): The top-k accuracy value.
            all_targets (torch.Tensor): The tensor containing all target values.
            samples (torch.Tensor): The tensor containing all sample values.
            top_k_scores (torch.Tensor): The tensor containing top-k scores.
            top_k_predictions (torch.Tensor): The tensor containing top-k predictions.
            class_names (dict): A dictionary containing class names.

        Returns:
            None
        """
        self.k = k
        self.top_1_acc = top_1_acc
        self.top_k_acc = top_k_acc
        self.all_targets = all_targets
        self.samples = samples
        self.top_k_scores = top_k_scores
        self.top_k_predictions = top_k_predictions
        self.class_names = class_names

    def compute_top_1_scores(self) -> Tuple[np.ndarray, float, float, float]:
        """
        Compute the top 1 scores for the given predictions and true targets.

        Returns:
            Tuple[np.ndarray, float, float, float]: A tuple containing confusion matrix, precision, recall, and F1 score.
        """
        top_1_predictions = self.top_k_predictions[:, 0]
        true_targets = self.all_targets

        cm = confusion_matrix(true_targets, top_1_predictions)
        precision = precision_score(true_targets, top_1_predictions, average="macro")
        recall = recall_score(true_targets, top_1_predictions, average="macro")
        f1 = f1_score(true_targets, top_1_predictions, average="macro")
        return cm, precision, recall, f1

    def compute_top_k_scores(self) -> Tuple[np.ndarray, float, float, float]:
        """
        Compute the top k scores based on the true targets and processed predictions.

        Returns:
            Tuple[np.ndarray, float, float, float]: A tuple containing the confusion matrix, precision score, recall score, and F1 score.
        """
        true_targets = self.all_targets
        processed_predictions = []

        for i in range(self.all_targets.shape[0]):
            if self.all_targets[i].item() in self.top_k_predictions[i].numpy().tolist():
                processed_predictions.append(self.all_targets[i].item())
            else:
                processed_predictions.append(self.top_k_predictions[i, 0].item())
        true_predictions = np.array(processed_predictions)

        cm = confusion_matrix(true_targets.numpy(), true_predictions)
        precision = precision_score(true_targets, true_predictions, average="macro")
        recall = recall_score(true_targets, true_predictions, average="macro")
        f1 = f1_score(true_targets, true_predictions, average="macro")
        return cm, precision, recall, f1
