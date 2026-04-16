import os

import pandas as pd
import torch

from Dataset import dataset
from UNITO_Model import UNITO
from Utils_Predict import *
from Utils_Train import is_cuda_device, predict_visualization, resolve_loader_settings
from torch.utils.data import DataLoader


def UNITO_gating(
    model_path,
    x_axis,
    y_axis,
    gate,
    path_raw,
    num_workers,
    device,
    save_prediction_path,
    dest,
    seq=False,
    gate_pre=None,
):
    """
    Perform UNITO auto-gating with a saved PyTorch model.
    """
    model = UNITO().to(device)
    model.load_state_dict(torch.load(os.path.join(model_path), map_location=device))
    model.eval()

    path_val = pd.read_csv(f"{dest}/Data/Data_{gate}/pred/subj.csv")
    val_ds = dataset(path_val, cache_in_memory=True)

    resolved_workers, pin_memory, persistent_workers = resolve_loader_settings(
        device,
        requested_workers=num_workers,
    )
    val_loader_kwargs = {
        "batch_size": path_val.shape[0],
        "num_workers": resolved_workers,
        "pin_memory": pin_memory,
        "shuffle": False,
    }
    if resolved_workers > 0:
        val_loader_kwargs["persistent_workers"] = persistent_workers
    val_loader = DataLoader(val_ds, **val_loader_kwargs)

    preds_list, y_val_list, x_list, subj_list = predict_visualization(
        val_loader,
        model,
        device=device,
        use_amp=is_cuda_device(device),
    )

    for ind in range(path_val.shape[0]):
        data_df_pred, subj_path = mask_to_gate(
            y_val_list,
            preds_list,
            x_list,
            subj_list,
            x_axis,
            y_axis,
            gate,
            gate_pre,
            path_raw,
            save_prediction_path,
            dest,
            worker=0,
            idx=ind,
            seq=seq,
        )

    print("UNITO prediction finished")

    return data_df_pred


def clean_val_path(subj_path, gate):
    """
    Formulate the path for reading purpose.
    """
    if "Raw" in subj_path:
        substring = os.path.join(f"./Data_image/Data_{gate}/Raw_Numpy/")
    else:
        substring = os.path.join(f"./Data_image/Data_{gate}/Mask_Numpy/")
    subj_path = subj_path.split(substring)[1]
    substring = ".csv.npy"
    subj_path = subj_path.split(substring)[0]
    return subj_path
