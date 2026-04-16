import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from UNITO_Model import UNITO
from Utils_Train import configure_runtime, fit_model, get_loaders, is_cuda_device


def export_model_onnx(model, onnx_path, input_shape=(1, 1, 101, 101), opset_version=17):
    """
    Export a trained UNITO model to ONNX with a dynamic batch axis.
    """
    export_model = UNITO(in_channels=1, out_channels=1)
    export_model.load_state_dict(model.state_dict())
    export_model.eval()

    dummy_input = torch.randn(*input_shape, dtype=torch.float32)
    torch.onnx.export(
        export_model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )


def train(
    gate,
    learning_rate,
    device,
    batch_size,
    epochs,
    n_worker,
    dest,
    use_amp=None,
    export_onnx=True,
    onnx_opset=17,
    cache_in_memory=True,
):
    """
    Train UNITO using the selected hyperparameters, then save .pt and .onnx models.
    """
    configure_runtime(device)
    use_amp = is_cuda_device(device) if use_amp is None else bool(use_amp and is_cuda_device(device))

    path_train = pd.read_csv(f"{dest}/Data/Data_{gate}/train/subj.csv")
    train_loader, _ = get_loaders(
        path_train,
        None,
        batch_size=batch_size,
        num_workers=n_worker,
        device=device,
        cache_in_memory=cache_in_memory,
    )

    model = UNITO(in_channels=1, out_channels=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    training_summary = fit_model(
        train_loader,
        model,
        optimizer,
        loss_fn,
        device=device,
        epochs=epochs,
        val_loader=None,
        use_amp=use_amp,
    )

    os.makedirs(f"{dest}/model", exist_ok=True)
    pt_path = os.path.join(f"{dest}/model", gate + "_model.pt")
    torch.save(model.state_dict(), pt_path)

    onnx_path = None
    if export_onnx:
        onnx_path = os.path.join(f"{dest}/model", gate + "_model.onnx")
        export_model_onnx(model, onnx_path, opset_version=onnx_opset)

    return {
        "pt_path": pt_path,
        "onnx_path": onnx_path,
        "training": training_summary,
    }
