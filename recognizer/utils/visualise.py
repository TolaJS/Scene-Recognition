import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Tuple
from random import randint
from torch.utils.data import DataLoader
from .constants import Constants
from .types import EvalResults


class Visualise:
    @staticmethod
    def imshow(img: np.ndarray, ax, title: str = None) -> None:
        """
        Denormalise the image from its normalised form based on ImageNet dataset's mean and standard deviation.

        Parameters:
            img (np.ndarray): The image to display.
            ax: The axis to display the image on.
            title (str): The title of the image (optional).

        Returns:
            None
        """

        # Denormalise the image from its normalised form based on ImageNet dataset's mean and standard deviation
        mean = np.array(Constants.IMAGE_NET_MEANS).reshape((1, 1, 3))
        std = np.array(Constants.IMAGE_NET_STDS).reshape((1, 1, 3))
        img = img.transpose((1, 2, 0))
        img = (std * img) + mean
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        if title is not None:
            ax.text(0.5, -0.1, title, transform=ax.transAxes, ha="center", va="top")
            # ax.set_title(title)

        ax.axis("off")

    @staticmethod
    def get_one_image_per_class(dataloader: DataLoader, class_names: dict) -> dict:
        """
        Get one image per class from the given dataloader based on the provided class names.

        Parameters:
            dataloader (DataLoader): The data loader containing images and labels.
            class_names (dict): A dictionary mapping class labels to class names.

        Returns:
            dict: A dictionary containing one image per class.
        """
        class_images = {}
        for images, labels in dataloader:
            for image, label in zip(images, labels):
                class_label = class_names[int(label)]
                if class_label not in class_images:
                    class_images[class_label] = image.numpy()
                if len(class_images) == len(
                    class_names
                ):  # Stop once we have one image per class
                    return class_images
        return class_images

    @staticmethod
    def display_images_grid(class_images: dict, class_names: dict) -> None:
        """
        Generate a grid of images for visualization.

        Parameters:
            class_images (dict): A dictionary containing images for each class.
            class_names (dict): A dictionary mapping class IDs to class names.

        Returns:
            None
        """
        rows = len(class_names) // 4
        fig, axes = plt.subplots(nrows=rows, ncols=4, figsize=(20, 20))
        for ax, class_name, class_id in zip(
            axes.flat, class_names.values(), class_names.keys()
        ):
            img = class_images[class_name]
            Visualise.imshow(img, ax, title=f"{class_name} ({class_id})")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def display_sample_per_class(loader: DataLoader) -> None:
        """
        Generate a visual display of one sample image per class in the DataLoader.

        Parameters:
            loader (DataLoader): The DataLoader containing the dataset.

        Returns:
            None
        """
        if hasattr(loader.dataset, "dataset"):
            class_names = {
                value: key
                for key, value in loader.dataset.dataset.class_to_idx.items()
            }
        else:
            class_names = {
                value: key for key, value in loader.dataset.class_to_idx.items()
            }
        class_images = Visualise.get_one_image_per_class(loader, class_names)
        Visualise.display_images_grid(class_images, class_names)

    @staticmethod
    def plot_confusion_matrix(
        cm: np.ndarray,
        cmap: str = "Greens",
        title: str = "Top-1 Confusion matrix",
        size: Tuple[int, int] = (20, 20),
    ) -> None:
        """
        A method to plot a confusion matrix using seaborn and matplotlib.

        Parameters:
            cm (np.ndarray): The confusion matrix to be plotted.
            cmap (str): The color map for the heatmap. Default is "Greens".
            title (str): The title of the plot. Default is "Top-1 Confusion matrix".
            size (Tuple[int, int]): The size of the plot. Default is (20, 20).

        Returns:
            None
        """
        plt.figure(figsize=size)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap=cmap, xticklabels=True, yticklabels=True
        )
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(title)
        plt.show()

    @staticmethod
    def show_results(
        eval_results: EvalResults, n: int = 10, fig_size: Tuple[int, int] = (12, 50)
    ) -> None:
        """
        Show the results of the evaluation with visual samples and corresponding predictions.

        Parameters:
            eval_results (EvalResults): The evaluation results object containing samples, predictions, and class names.
            n (int): Number of visual samples to display (default is 10).
            fig_size (Tuple[int, int]): Size of the figure for displaying the visual samples (default is (12, 50)).

        Returns:
            None
        """
        rows = n // 2
        fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=fig_size)

        (
            actual_class_name,
            actual_class_id,
            images,
            predicted_class_id,
            predicted_class_name,
            predicted_score,
            titles,
        ) = ([], [], [], [], [], [], [])

        random_visual_sample = [randint(0, len(eval_results.samples) - 1) for _ in range(n)]

        for i in random_visual_sample:
            _ac_id = eval_results.all_targets[i].item()
            _pc_id = eval_results.top_k_predictions[i, 0].item()
            actual_class_name.append(eval_results.class_names[_ac_id])
            actual_class_id.append(_ac_id)
            images.append(eval_results.samples[i].numpy())
            predicted_class_id.append(_pc_id)
            predicted_class_name.append(eval_results.class_names[_pc_id])
            predicted_score.append(eval_results.top_k_scores[i, 0].item())
            actual_info = f"""
            Prediction:
            ==========
            Actual: {eval_results.class_names[_ac_id]}
            Inferred: {eval_results.class_names[_pc_id]}
            
            Scores:
            =======
            
            """
            predicted_info = "\n".join(
                [
                    f"{eval_results.class_names[pred]}: {score:.2f}"
                    for pred, score in zip(
                        eval_results.top_k_predictions[i].numpy().tolist(),
                        eval_results.top_k_scores[i].numpy().tolist(),
                    )
                ]
            )
            titles.append("".join([actual_info, predicted_info]))

        iterable = zip(
            axes.flat,
            images,
            actual_class_name,
            actual_class_id,
            predicted_class_name,
            predicted_class_id,
            titles,
        )

        for (
            ax,
            img,
            actual_class_name,
            actual_class_id,
            predicted_class_name,
            predicted_class_id,
            title,
        ) in iterable:
            Visualise.imshow(img, ax, title=title)
        plt.tight_layout()
        plt.show()
