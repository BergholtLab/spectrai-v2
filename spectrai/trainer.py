"""Training loop, evaluation, and application.

Ported from the reference ``spectrai/trainer.py``.
"""

from __future__ import annotations

import copy
import datetime
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from .data import (
    prepare_data,
    prepare_dataloader,
    split_data,
)
from .networks.registry import edit_network, setup_network
from .utils import (
    AverageMeter,
    calc_psnr,
    calc_ssim,
    check_inputs,
    mixup_criterion,
    mixup_data,
)
from .transforms import prepare_transforms

__all__ = [
    "train_epoch",
    "train",
    "validate",
    "evaluate_pretrained",
    "apply_pretrained",
    "apply_net",
    "save_output",
    "load_pretrained",
    "initialise_network",
    "initialise_training_hyperparameters",
    "initialise_dataset",
]

VAL_AUG_OFF = {
    "horizontal_flip": 0, "vertical_flip": 0, "rotation": 0, "random_crop": 0,
    "spectral_shift": 0.0, "spectral_flip": 0, "spectral_background": 0, "mixup": 0,
}


# ---------------------------------------------------------------------------
# Initialisation (ported from the reference ``initialise.py``)
# ---------------------------------------------------------------------------


def _is_null(value) -> bool:
    """True if the value is a legacy null sentinel (real None or the string 'None')."""
    return value is None or (isinstance(value, str) and value == "None")


def _has_split(DataManager_Options) -> bool:
    """True if any of train/val/test split is set (and data is present)."""
    splits = (
        DataManager_Options.get("train_split"),
        DataManager_Options.get("val_split"),
        DataManager_Options.get("test_split"),
    )
    has_data = (
        DataManager_Options.get("train_input_data") is not None
        or DataManager_Options.get("train_target_data") is not None
    )
    return any(not _is_null(s) for s in splits) and has_data


def initialise_network(Training_Options, Task_Options, Network_Hyperparameters,
                       Training_Hyperparameters, DataManager_Options, net_state_dict):
    if _is_null(Training_Options["pretrained_network"]):
        net = setup_network(Task_Options, Network_Hyperparameters, Training_Hyperparameters, DataManager_Options)
        if not _is_null(net_state_dict):
            net.load_state_dict(net_state_dict)
        return net

    # Transfer learning / apply pre-trained
    if int(Training_Options.get("pretrained_classes", 0)) == int(Task_Options["classes"]):
        net = setup_network(Task_Options, Network_Hyperparameters, Training_Hyperparameters, DataManager_Options)
        pretrained = torch.load(Training_Options["pretrained_network"], map_location="cpu")
        net.load_state_dict(pretrained["network"])
    else:
        new_classes = int(Task_Options["classes"])
        pretrained_classes = int(Training_Options.get("pretrained_classes", 0))
        Task_Options = dict(Task_Options)
        Task_Options["classes"] = pretrained_classes
        net = setup_network(Task_Options, Network_Hyperparameters, Training_Hyperparameters, DataManager_Options)
        pretrained = torch.load(Training_Options["pretrained_network"], map_location="cpu")
        net.load_state_dict(pretrained["network"])
        Task_Options["classes"] = new_classes
        net = edit_network(net, Task_Options, Network_Hyperparameters, Training_Hyperparameters, DataManager_Options)
    return net


def initialise_training_hyperparameters(Training_Hyperparameters, net, dataloader):
    # Criterion
    if Training_Hyperparameters["criterion"] == "L1":
        criterion = nn.L1Loss()
    elif Training_Hyperparameters["criterion"] == "L2 / MSE":
        criterion = nn.MSELoss()
    elif Training_Hyperparameters["criterion"] == "Cross Entropy":
        criterion = nn.CrossEntropyLoss()
    else:  # Binary Cross Entropy
        criterion = nn.BCELoss()

    # Optimizer
    lr = Training_Hyperparameters["learning_rate"]
    if Training_Hyperparameters["optimizer"] == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=lr)
    elif Training_Hyperparameters["optimizer"] == "Adagrad":
        optimizer = optim.Adagrad(net.parameters(), lr=lr)
    elif Training_Hyperparameters["optimizer"] == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=lr)
    else:  # RMSprop
        optimizer = optim.RMSprop(net.parameters(), lr=lr)

    # Scheduler
    if Training_Hyperparameters["scheduler"] == "Step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
    elif Training_Hyperparameters["scheduler"] == "Multiplicative":
        scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 0.985)
    elif Training_Hyperparameters["scheduler"] == "Cyclic":
        scheduler = optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=lr / 100, max_lr=lr, mode="triangular2", cycle_momentum=False
        )
    elif Training_Hyperparameters["scheduler"] == "OneCycle" and dataloader is not None:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=len(dataloader),
            epochs=int(Training_Hyperparameters["epochs"]), cycle_momentum=False,
        )
    elif Training_Hyperparameters["scheduler"] == "ReduceOnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    else:  # Constant
        scheduler = None

    return criterion, optimizer, scheduler


def initialise_dataset(Task_Options, Network_Hyperparameters, Training_Hyperparameters,
                       Preprocessing, Data_Augmentation, DataManager_Options, apply=0):
    val_aug = dict(VAL_AUG_OFF)
    val_dmo = copy.deepcopy(DataManager_Options)
    val_dmo["shuffle"] = "False"

    def _build(dmo, aug):
        transforms = prepare_transforms(Task_Options, Network_Hyperparameters, Training_Hyperparameters,
                                        Preprocessing, aug, dmo)
        return transforms

    if _has_split(DataManager_Options):
        input_data, target_data, input_type, target_type, directory = prepare_data(
            DataManager_Options, Task_Options, "train", apply
        )
        data, input_type, target_type, directory = split_data(
            Task_Options, DataManager_Options, input_data, target_data, input_type, target_type, directory
        )
        tr_transforms = _build(DataManager_Options, Data_Augmentation)
        val_transforms = _build(val_dmo, val_aug)

        train_loader = (
            prepare_dataloader(Task_Options, Training_Hyperparameters, DataManager_Options,
                               data["train_input"], data["train_target"], input_type, target_type,
                               directory, tr_transforms, apply)
            if (data["train_input"] is not None or data["train_target"] is not None)
            else None
        )
        val_loader = (
            prepare_dataloader(Task_Options, Training_Hyperparameters, DataManager_Options,
                               data["val_input"], data["val_target"], input_type, target_type,
                               directory, val_transforms, apply)
            if (data["val_input"] is not None or data["val_target"] is not None)
            else None
        )
        test_loader = (
            prepare_dataloader(Task_Options, Training_Hyperparameters, DataManager_Options,
                               data["test_input"], data["test_target"], input_type, target_type,
                               directory, val_transforms, apply)
            if (data["test_input"] is not None or data["test_target"] is not None)
            else None
        )
    else:
        train_loader = val_loader = test_loader = None
        if not _is_null(DataManager_Options.get("train_input_data")) or not _is_null(DataManager_Options.get("train_target_data")):
            input_data, target_data, input_type, target_type, directory = prepare_data(
                DataManager_Options, Task_Options, "train", apply
            )
            train_loader = prepare_dataloader(
                Task_Options, Training_Hyperparameters, DataManager_Options,
                input_data, target_data, input_type, target_type, directory,
                _build(DataManager_Options, Data_Augmentation), apply,
            )
        if not _is_null(DataManager_Options.get("val_input_data")) or not _is_null(DataManager_Options.get("val_target_data")):
            input_data, target_data, input_type, target_type, directory = prepare_data(
                DataManager_Options, Task_Options, "val", apply
            )
            val_loader = prepare_dataloader(
                Task_Options, Training_Hyperparameters, DataManager_Options,
                input_data, target_data, input_type, target_type, directory,
                _build(val_dmo, val_aug), apply,
            )
        if not _is_null(DataManager_Options.get("test_input_data")) or not _is_null(DataManager_Options.get("test_target_data")):
            input_data, target_data, input_type, target_type, directory = prepare_data(
                DataManager_Options, Task_Options, "test", apply
            )
            test_loader = prepare_dataloader(
                Task_Options, Training_Hyperparameters, DataManager_Options,
                input_data, target_data, input_type, target_type, directory,
                _build(val_dmo, val_aug), apply,
            )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Training / validation steps
# ---------------------------------------------------------------------------


def train(Task_Options, Training_Hyperparameters, Data_Augmentation, net, criterion, optimizer,
          scheduler, dataloader, device):
    task = Task_Options["task"]
    losses = AverageMeter("Loss", ":.4e")
    if task == "Super-Resolution":
        psnr, ssim = AverageMeter("PSNR", ":.4f"), AverageMeter("SSIM", ":.4f")
    elif task == "Classification":
        acc = AverageMeter("Accuracy", ":.4f")

    end = time.time()
    metrics = {}
    for sample in dataloader:
        inputs = sample["input"].float().to(device)
        target = sample["target"]["data"]
        if task in ("Segmentation", "Classification"):
            target = target.long()
        else:
            target = target.float()
        target = target.to(device)

        if Data_Augmentation.get("mixup"):
            inputs, target_a, target_b, lam = mixup_data(inputs, target, alpha=0.2)
            output = net(inputs)
            loss = mixup_criterion(criterion, output, target_a, target_b, lam)
        else:
            output = net(inputs)
            loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if Training_Hyperparameters["scheduler"] in ("Cyclic", "OneCycle"):
            scheduler.step()

        losses.update(loss.item(), inputs.size(0))
        batch_time = time.time() - end
        end = time.time()
        metrics = {"loss": losses.avg, "time": batch_time}

        if task == "Super-Resolution":
            psnr.update(calc_psnr(output, target), inputs.size(0))
            ssim.update(calc_ssim(output, target), inputs.size(0))
            metrics["psnr"] = psnr.avg
            metrics["ssim"] = ssim.avg
        elif task == "Classification":
            _, preds = torch.max(output, 1)
            acc_batch = torch.sum(preds == target.data).item() / inputs.size(0)
            acc.update(acc_batch, inputs.size(0))
            metrics["accuracy"] = acc.avg

    return metrics


def validate(Task_Options, net, criterion, dataloader, device):
    task = Task_Options["task"]
    losses = AverageMeter("Loss", ":.4e")
    if task == "Super-Resolution":
        psnr, ssim = AverageMeter("PSNR", ":.4f"), AverageMeter("SSIM", ":.4f")
    elif task == "Classification":
        acc = AverageMeter("Accuracy", ":.4f")

    net.eval()
    metrics = {}
    with torch.no_grad():
        end = time.time()
        for sample in dataloader:
            inputs = sample["input"].float().to(device)
            target = sample["target"]["data"]
            if task in ("Segmentation", "Classification"):
                target = target.long()
            else:
                target = target.float()
            target = target.to(device)

            output = net(inputs)
            loss = criterion(output, target)
            losses.update(loss.item(), inputs.size(0))

            batch_time = time.time() - end
            end = time.time()
            metrics = {"loss": losses.avg, "time": batch_time}

            if task == "Super-Resolution":
                psnr.update(calc_psnr(output, target), inputs.size(0))
                ssim.update(calc_ssim(output, target), inputs.size(0))
                metrics["psnr"] = psnr.avg
                metrics["ssim"] = ssim.avg
            elif task == "Classification":
                _, preds = torch.max(output, 1)
                acc_batch = torch.sum(preds == target.data).item() / inputs.size(0)
                acc.update(acc_batch, inputs.size(0))
                metrics["accuracy"] = acc.avg

    return metrics


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def train_epoch(Training_Options, Task_Options, Network_Hyperparameters, Training_Hyperparameters,
                Preprocessing, Data_Augmentation, DataManager_Options, net_state_dict,
                optimizer_state_dict, scheduler_state_dict, epochs, results_preview=0,
                current_epoch=1, max_epochs=10, save_frequency=0, verbose=False):
    check_inputs(Training_Options, Task_Options, Network_Hyperparameters, Training_Hyperparameters,
                 Preprocessing, Data_Augmentation, DataManager_Options)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if DataManager_Options.get("seed") not in (None, "None"):
        random.seed(int(DataManager_Options["seed"]))
        torch.manual_seed(int(DataManager_Options["seed"]))
        cudnn.deterministic = True

    net = initialise_network(Training_Options, Task_Options, Network_Hyperparameters,
                             Training_Hyperparameters, DataManager_Options, net_state_dict)
    net.to(device)

    train_loader, val_loader, test_loader = initialise_dataset(
        Task_Options, Network_Hyperparameters, Training_Hyperparameters,
        Preprocessing, Data_Augmentation, DataManager_Options,
    )

    criterion, optimizer, scheduler = initialise_training_hyperparameters(
        Training_Hyperparameters, net, train_loader
    )

    if not _is_null(optimizer_state_dict):
        optimizer.load_state_dict(optimizer_state_dict)
    if scheduler is not None and not _is_null(scheduler_state_dict):
        scheduler.load_state_dict(scheduler_state_dict)

    date = datetime.datetime.now().strftime("%Y_%m_%d")
    log_dir = os.path.join(os.getcwd(), "tensorboard", f"{date}_{Task_Options['task']}_{Network_Hyperparameters['network']}")
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(int(epochs)):
        train_metrics = None
        if train_loader is not None:
            train_metrics = train(Task_Options, Training_Hyperparameters, Data_Augmentation,
                                 net, criterion, optimizer, scheduler, train_loader, device)
            if verbose:
                print(f"Epoch: {epoch} Train: {train_metrics}")
                for key, val in train_metrics.items():
                    if isinstance(val, float):
                        writer.add_scalar(f"Train/{key}", val, epoch)

        val_metrics = None
        if val_loader is not None:
            val_metrics = validate(Task_Options, net, criterion, val_loader, device)
            if verbose:
                print(f"Epoch: {epoch} Validation: {val_metrics}")
                for key, val in val_metrics.items():
                    if isinstance(val, float):
                        writer.add_scalar(f"Validation/{key}", val, epoch)

        test_metrics = None
        if test_loader is not None:
            test_metrics = validate(Task_Options, net, criterion, test_loader, device)

        if Training_Hyperparameters["scheduler"] in ("Step", "Multiplicative"):
            scheduler.step()
        elif Training_Hyperparameters["scheduler"] == "ReduceOnPlateau":
            if val_metrics is not None:
                scheduler.step(val_metrics["loss"])
            elif train_metrics is not None:
                scheduler.step(train_metrics["loss"])
            else:
                raise ValueError("ReduceOnPlateau LR should only be used when training, ideally with a validation dataset")

        current_epoch += 1
        if (save_frequency and current_epoch % save_frequency == 0) or current_epoch == max_epochs:
            save_output(Network_Hyperparameters, Task_Options, Training_Hyperparameters,
                        DataManager_Options, net.state_dict(), optimizer.state_dict(),
                        scheduler_state_dict, criterion, train_metrics, val_metrics, test_metrics)

    writer.close()

    classes = int(Task_Options["classes"])
    output = {
        "network": net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler_state_dict,
        "criterion": criterion,
        "classes": classes,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    return output


def save_output(Network_Hyperparameters, Task_Options, Training_Hyperparameters, DataManager_Options,
                net, optimizer, scheduler, criterion, train_metrics, val_metrics, test_metrics,
                out_dir: str | None = None):
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "saved_models")
    os.makedirs(out_dir, exist_ok=True)

    date = datetime.datetime.now().strftime("%Y_%m_%d")
    save_name = f"{date}_{Task_Options['task']}_{Network_Hyperparameters['network']}.pt"

    payload = {
        "network": net, "optimizer": optimizer, "scheduler": scheduler, "criterion": criterion,
        "train_metrics": train_metrics, "val_metrics": val_metrics, "test_metrics": test_metrics,
        "architecture": Network_Hyperparameters["network"], "task": Task_Options["task"],
        "channels": int(Training_Hyperparameters["spectrum_length"]),
        "normalization": Network_Hyperparameters["normalization"],
        "activation": Network_Hyperparameters["activation"],
        "classes": int(Task_Options["classes"]),
        "input_image_size": int(Training_Hyperparameters["input_image_size"]),
        "target_image_size": int(Training_Hyperparameters["target_image_size"]),
        "data_format": DataManager_Options["data_format"],
    }
    torch.save(payload, os.path.join(out_dir, save_name))
    return os.path.join(out_dir, save_name)


def load_pretrained(pretrained_network: str) -> dict:
    data = torch.load(pretrained_network, map_location="cpu")
    return {
        "architecture": data["architecture"], "task": data["task"], "channels": data["channels"],
        "normalization": data["normalization"], "activation": data["activation"],
        "classes": data["classes"], "input_image_size": data["input_image_size"],
        "target_image_size": data["target_image_size"], "data_format": data["data_format"],
    }


def evaluate_pretrained(Training_Options, Task_Options, Network_Hyperparameters, Training_Hyperparameters,
                        Preprocessing, Data_Augmentation, DataManager_Options, net_state_dict, results_preview=0):
    check_inputs(Training_Options, Task_Options, Network_Hyperparameters, Training_Hyperparameters,
                 Preprocessing, Data_Augmentation, DataManager_Options)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if DataManager_Options.get("seed") not in (None, "None"):
        random.seed(int(DataManager_Options["seed"]))
        torch.manual_seed(int(DataManager_Options["seed"]))
        cudnn.deterministic = True

    net = initialise_network(Training_Options, Task_Options, Network_Hyperparameters,
                             Training_Hyperparameters, DataManager_Options, net_state_dict)
    net.to(device)

    _, _, test_loader = initialise_dataset(Task_Options, Network_Hyperparameters, Training_Hyperparameters,
                                           Preprocessing, Data_Augmentation, DataManager_Options)
    criterion, _, _ = initialise_training_hyperparameters(
        Training_Hyperparameters, net, test_loader
    )

    test_metrics = None
    if test_loader is not None:
        test_metrics = validate(Task_Options, net, criterion, test_loader, device)

    return {"test_metrics": test_metrics}


def apply_net(Task_Options, Network_Hyperparameters, net, dataloader, device, out_dir: str | None = None):
    architecture = Network_Hyperparameters["network"]
    task = Task_Options["task"]
    date = datetime.datetime.now().strftime("%Y_%m_%d")
    save_name = f"{date}_{task}_{architecture}"
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "apply_outputs", save_name)
    os.makedirs(out_dir, exist_ok=True)

    net.eval()
    with torch.no_grad():
        for data in dataloader:
            names = data["name"]
            inputs = data["input"].float().to(device)
            outputs = net(inputs)
            net_out = outputs.detach().cpu().numpy()
            for j in range(net_out.shape[0]):
                output_data = np.squeeze(net_out[j])
                file_name = f"{names[j]}_output.npy"
                np.save(os.path.join(out_dir, file_name), output_data)


def apply_pretrained(Training_Options, Task_Options, Network_Hyperparameters, Training_Hyperparameters,
                     Preprocessing, Data_Augmentation, DataManager_Options, net_state_dict,
                     results_preview=0, apply=1, out_dir: str | None = None):
    check_inputs(Training_Options, Task_Options, Network_Hyperparameters, Training_Hyperparameters,
                 Preprocessing, Data_Augmentation, DataManager_Options)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = initialise_network(Training_Options, Task_Options, Network_Hyperparameters,
                             Training_Hyperparameters, DataManager_Options, net_state_dict)
    net.to(device)

    _, _, test_loader = initialise_dataset(Task_Options, Network_Hyperparameters, Training_Hyperparameters,
                                           Preprocessing, Data_Augmentation, DataManager_Options, apply)

    if test_loader is not None:
        apply_net(Task_Options, Network_Hyperparameters, net, test_loader, device, out_dir=out_dir)

    return {"applied": test_loader is not None}
