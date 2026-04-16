import os

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from Dataset import dataset


def is_cuda_device(device):
    return str(device).startswith("cuda")


def configure_runtime(device):
    if is_cuda_device(device):
        torch.backends.cudnn.benchmark = True


def resolve_loader_settings(device, requested_workers=None):
    if requested_workers is None:
        requested_workers = min(4, os.cpu_count() or 0) if is_cuda_device(device) else 0

    requested_workers = max(int(requested_workers), 0)
    pin_memory = is_cuda_device(device)
    persistent_workers = pin_memory and requested_workers > 0

    return requested_workers, pin_memory, persistent_workers


def build_data_loader(data_source, batch_size, shuffle, num_workers, pin_memory, persistent_workers):
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers

    return DataLoader(data_source, **loader_kwargs)


def get_loaders(path_train, path_test, batch_size, num_workers=None, device="cpu", cache_in_memory=True):
    """
    Prepare train and validation data loaders.
    """
    resolved_workers, pin_memory, persistent_workers = resolve_loader_settings(
        device,
        requested_workers=num_workers,
    )

    train_ds = dataset(path_train, cache_in_memory=cache_in_memory)
    train_loader = build_data_loader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=resolved_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    test_loader = None
    if path_test is not None:
        test_ds = dataset(path_test, cache_in_memory=cache_in_memory)
        test_loader = build_data_loader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=resolved_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    return train_loader, test_loader


def train_epoch(loader, model, optimizer, loss_fn, device, scaler=None, use_amp=False):
    """
    Perform one epoch of training.
    """
    loss_total = 0.0
    batch_count = 0

    for data, target, _ in loader:
        data = data.to(device=device, non_blocking=True)
        target = target.unsqueeze(1).to(device=device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            predictions = model(data)
            loss = loss_fn(predictions, target)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_total += loss.item()
        batch_count += 1

    if batch_count == 0:
        return 0.0

    return loss_total / batch_count


def check_accuracy(loader, model, device="cpu", use_amp=False):
    """
    Calculate accuracy and Dice score for the assigned data loader.
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0.0
    model.eval()

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device=device, non_blocking=True)
            y = y.unsqueeze(1).to(device=device, non_blocking=True)

            with autocast(enabled=use_amp):
                preds = torch.sigmoid(model(x))

            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum().item()
            num_pixels += preds.numel()
            dice_score += (
                (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            ).item()

    model.train()

    accuracy = (num_correct / num_pixels * 100.0) if num_pixels else 0.0
    average_dice = dice_score / len(loader) if len(loader) else 0.0

    return accuracy, average_dice


def fit_model(
    train_loader,
    model,
    optimizer,
    loss_fn,
    device,
    epochs,
    val_loader=None,
    val_every=1,
    patience=None,
    use_amp=False,
):
    """
    Shared fit loop used for both tuning and final training.
    """
    use_amp = bool(use_amp and is_cuda_device(device))
    scaler = GradScaler(enabled=use_amp)

    best_dice = None
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    last_train_loss = 0.0
    epochs_ran = 0

    for epoch in range(1, epochs + 1):
        last_train_loss = train_epoch(
            train_loader,
            model,
            optimizer,
            loss_fn,
            device,
            scaler=scaler,
            use_amp=use_amp,
        )
        epochs_ran = epoch

        should_validate = val_loader is not None and (epoch % val_every == 0 or epoch == epochs)
        if not should_validate:
            continue

        _, val_dice = check_accuracy(
            val_loader,
            model,
            device=device,
            use_amp=use_amp,
        )

        if best_dice is None or val_dice > best_dice:
            best_dice = val_dice
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1
            if patience is not None and epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_dice": best_dice,
        "best_epoch": best_epoch,
        "epochs_ran": epochs_ran,
        "last_train_loss": last_train_loss,
    }


def predict_visualization(loader, model, device="cpu", use_amp=False):
    """
    Predict masks for a prepared loader.
    """
    model.eval()
    preds_list = []
    y_list = []
    x_list = []
    subj_list = []

    with torch.no_grad():
        for x, y, subj in loader:
            x = x.to(device=device, non_blocking=True)
            with autocast(enabled=use_amp):
                preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            preds_list.append(preds.cpu())
            y_list.append(y.unsqueeze(1))
            x_list.append(x.cpu())
            subj_list.append(subj)

    return preds_list, y_list, x_list, subj_list
