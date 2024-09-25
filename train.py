import torch
import logging

from recognizer.data import ProjectDataset
from recognizer.model import ResNetFineTunedClassifier
from recognizer.trainer import TorchRunner

# configuring logging level
logging.basicConfig(level=logging.INFO)


def run(
    training_dataset_path: str, testing_dataset_path: str, _device: torch.device = None
):
    """
    A function that runs the training and evaluation process of a classifier model.

    Parameters:
    - training_dataset_path: str, the path to the training dataset
    - testing_dataset_path: str, the path to the testing dataset
    - _device: torch.device, the device to run the training on (default is None)

    Returns:
    No direct return, but saves checkpoints and logs evaluation results.
    """
    run_id = 25
    batch_size = 512
    lr = 0.0001
    gamma = 0.90
    weight_decay = 0.0005
    epochs = 40

    train_loader, val_loader, test_loader = ProjectDataset.get_loaders(
        training_dataset_path=training_dataset_path,
        testing_dataset_path=testing_dataset_path,
        batch_size=batch_size,
    )

    classifier = ResNetFineTunedClassifier(freeze_pretrained_weights=False)
    TorchRunner.get_summary(classifier, (3, 244, 244))
    params_1x = [
        param for name, param in classifier.named_parameters() if "fc" not in str(name)
    ]

    # optimiser = torch.optim.Adam(classifier.parameters(), lr=0.001, weight_decay=0.0005)
    optimiser = torch.optim.Adam(
        [
            {"params": params_1x},
            {"params": classifier.model.fc.parameters(), "lr": 0.01},
        ],
        lr=lr,
        weight_decay=weight_decay,
    )
    exponential_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimiser, gamma=gamma
    )

    # plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimiser, mode="min", factor=0.2, patience=10, min_lr=0.000001
    # )

    TorchRunner.train(
        train_loader=train_loader,
        val_loader=val_loader,
        model=classifier,
        loss_fn=torch.nn.functional.cross_entropy,
        optimizer=optimiser,
        scheduler=exponential_scheduler,
        epochs=epochs,
        device=_device,
        runs_folder="./checkpoints/runs",
    )

    eval_results = TorchRunner.evaluate(
        loader=val_loader, model=classifier, k=5, device=device
    )
    top_1_cm, top_1_precision, top_1_recall, top_1_f1 = eval_results.compute_top_1_scores()
    logging.info(f"\nTop-1: {run_id}\nPrecision: {top_1_precision}\nRecall: {top_1_recall}\nF1: {top_1_f1}")

    top_k_cm, top_k_precision, top_k_recall, top_k_f1 = eval_results.compute_top_k_scores()
    logging.info(f"\nTop-5: {run_id}\nPrecision: {top_k_precision}\nRecall: {top_k_recall}\nF1: {top_k_f1}")
    classifier.save(f"./checkpoints/models/{run_id}.safetensors")


if __name__ == "__main__":
    # picking appropriate device to train the model
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # dataset related config
    datasets_dir = "./data"
    training_dataset = f"{datasets_dir}/train"
    testing_dataset = f"{datasets_dir}/test"
    run(training_dataset, testing_dataset, device)
