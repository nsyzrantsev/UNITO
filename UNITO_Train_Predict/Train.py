import inspect
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from UNITO_Model import UNITO
from Utils_Train import configure_runtime, fit_model, get_loaders, is_cuda_device


def build_onnx_metadata(gate_name, x_axis, y_axis, parent_gate=None):
    """
    Build ONNX metadata expected by downstream cytiq consumers.
    """
    metadata = {
        "cytiq.gate_name": str(gate_name),
        "cytiq.x_axis": str(x_axis),
        "cytiq.y_axis": str(y_axis),
    }

    if parent_gate is not None:
        parent_gate = str(parent_gate).strip()
        if parent_gate:
            metadata["cytiq.parent_gate"] = parent_gate

    return metadata


def export_model_onnx(
    model,
    onnx_path,
    gate_name,
    x_axis,
    y_axis,
    parent_gate=None,
    input_shape=(1, 1, 101, 101),
    opset_version=18,
):
    """
    Export a trained UNITO model to a single-file ONNX with cytiq metadata.
    """
    export_model = UNITO(in_channels=1, out_channels=1)
    export_model.load_state_dict(model.state_dict())
    export_model.eval()
    import onnx

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(*input_shape, dtype=torch.float32)
    export_signature = inspect.signature(torch.onnx.export)
    export_kwargs = {
        "export_params": True,
        "opset_version": opset_version,
        "input_names": ["input"],
        "output_names": ["logits"],
        "dynamic_axes": {
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    }
    if "external_data" in export_signature.parameters:
        export_kwargs["external_data"] = False
    elif "use_external_data_format" in export_signature.parameters:
        export_kwargs["use_external_data_format"] = False

    metadata = build_onnx_metadata(gate_name, x_axis, y_axis, parent_gate=parent_gate)
    with tempfile.TemporaryDirectory(dir=onnx_path.parent) as temp_dir:
        exported_onnx_path = Path(temp_dir) / onnx_path.name
        torch.onnx.export(
            export_model,
            dummy_input,
            str(exported_onnx_path),
            **export_kwargs,
        )

        exported_model = onnx.load_model(str(exported_onnx_path), load_external_data=True)
        onnx.helper.set_model_props(exported_model, metadata)
        onnx.save_model(exported_model, str(onnx_path), save_as_external_data=False)


def copy_model_artifacts(saved_models, export_dir):
    """
    Copy saved model artifacts to an external directory such as Google Drive.
    """
    if not export_dir:
        return {}

    os.makedirs(export_dir, exist_ok=True)
    copied_paths = {}

    for path_key in ["pt_path", "onnx_path"]:
        source_path = saved_models.get(path_key)
        if source_path is None:
            continue

        target_path = os.path.join(export_dir, os.path.basename(source_path))
        shutil.copy2(source_path, target_path)
        copied_paths[path_key] = target_path

    return copied_paths


def train(
    gate,
    x_axis,
    y_axis,
    learning_rate,
    device,
    batch_size,
    epochs,
    n_worker,
    dest,
    use_amp=None,
    export_onnx=True,
    onnx_opset=18,
    cache_in_memory=True,
    parent_gate=None,
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
        export_model_onnx(
            model,
            onnx_path,
            gate_name=gate,
            x_axis=x_axis,
            y_axis=y_axis,
            parent_gate=parent_gate,
            opset_version=onnx_opset,
        )

    return {
        "pt_path": pt_path,
        "onnx_path": onnx_path,
        "training": training_summary,
    }
