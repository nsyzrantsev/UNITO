import pandas as pd
import torch.nn as nn
import torch.optim as optim

from UNITO_Model import UNITO
from Utils_Train import configure_runtime, fit_model, get_loaders, is_cuda_device


def tune(
    gate,
    hyperparameter_set,
    device,
    max_epochs,
    n_worker,
    dest,
    val_every=5,
    patience=3,
    use_amp=None,
    cache_in_memory=True,
):
    """
    Tune learning rate and batch size using a held-out validation split.
    """
    configure_runtime(device)
    use_amp = is_cuda_device(device) if use_amp is None else bool(use_amp and is_cuda_device(device))

    best_lr, best_bs = hyperparameter_set[0]
    best_dice = float("-inf")

    path = pd.read_csv(f"{dest}/Data/Data_{gate}/train/subj.csv")
    if path.shape[0] < 2:
        path_train = path.reset_index(drop=True)
        path_test = path.reset_index(drop=True)
    else:
        cutoff = max(1, int(path.shape[0] * 7 / 8))
        cutoff = min(cutoff, path.shape[0] - 1)
        path_train = path.iloc[:cutoff].reset_index(drop=True)
        path_test = path.iloc[cutoff:].reset_index(drop=True)

    for learning_rate, batch_size in hyperparameter_set:
        model = UNITO(in_channels=1, out_channels=1).to(device)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_loader, test_loader = get_loaders(
            path_train,
            path_test,
            batch_size=batch_size,
            num_workers=n_worker,
            device=device,
            cache_in_memory=cache_in_memory,
        )

        fit_summary = fit_model(
            train_loader,
            model,
            optimizer,
            loss_fn,
            device=device,
            epochs=max_epochs,
            val_loader=test_loader,
            val_every=val_every,
            patience=patience,
            use_amp=use_amp,
        )

        current_dice = fit_summary["best_dice"] if fit_summary["best_dice"] is not None else 0.0
        if current_dice > best_dice:
            best_dice = current_dice
            best_lr = learning_rate
            best_bs = batch_size

    return best_lr, best_bs
