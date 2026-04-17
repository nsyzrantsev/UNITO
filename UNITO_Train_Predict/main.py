import os
import random
import warnings

import numpy as np
import pandas as pd
import torch

from Data_Preprocessing import process_table, train_test_val_split
from Predict import UNITO_gating, evaluation
from Train import copy_model_artifacts, train
from Validation_Recon_Plot_Single import plot_all
from hyperparameter_tunning import tune

warnings.filterwarnings("ignore")

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

gating = pd.read_csv("./gating_structure.csv")
gate_pre_list = list(gating.Parent_Gate)
gate_pre_list[0] = None
gate_list = list(gating.Gate)
x_axis_list = list(gating.X_axis)
y_axis_list = list(gating.Y_axis)
path2_lastgate_pred_list = ["./prediction/" for _ in range(len(gate_list))]
path2_lastgate_pred_list[0] = "./Raw_Data_pred/"

if torch.cuda.is_available():
    device = "cuda"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

n_worker = None
tune_epochs = 60
final_epochs = 150
val_every = 5
patience = 3
convex = True
save_png = False
force_rebuild = False
export_onnx = True
onnx_opset = 18
model_export_dir = None

hyperparameter_set = [
    [1e-3, 16],
    [3e-4, 16],
    [1e-3, 32],
    [3e-4, 32],
]

dest = "."
save_data_img_path = f"{dest}/Data"
save_figure_path = f"{dest}/figures"
save_model_path = f"{dest}/model"
save_prediction_path = f"{dest}/prediction"

for output_dir in [
    save_data_img_path,
    save_figure_path,
    save_model_path,
    save_prediction_path,
]:
    os.makedirs(output_dir, exist_ok=True)

hyperparameter_df = pd.DataFrame(columns=["gate", "learning_rate", "batch_size"])

for gate_pre, gate, x_axis, y_axis, path_raw in zip(
    gate_pre_list,
    gate_list,
    x_axis_list,
    y_axis_list,
    path2_lastgate_pred_list,
):
    print(f"start UNITO for {gate}")

    train_path = "./Raw_Data_train"
    process_table(
        x_axis,
        y_axis,
        gate_pre,
        gate,
        train_path,
        convex=convex,
        seq=(gate_pre is not None),
        dest=dest,
        save_png=save_png,
        force_rebuild=force_rebuild,
    )
    train_test_val_split(gate, train_path, dest, "train")

    best_lr, best_bs = tune(
        gate,
        hyperparameter_set,
        device,
        tune_epochs,
        n_worker,
        dest,
        val_every=val_every,
        patience=patience,
    )
    hyperparameter_df.loc[len(hyperparameter_df)] = [gate, best_lr, best_bs]

    saved_models = train(
        gate,
        best_lr,
        device,
        best_bs,
        final_epochs,
        n_worker,
        dest,
        export_onnx=export_onnx,
        onnx_opset=onnx_opset,
        x_axis=x_axis,
        y_axis=y_axis,
        parent_gate=gate_pre,
    )
    print(f"Saved PyTorch model: {saved_models['pt_path']}")
    if saved_models["onnx_path"] is not None:
        print(f"Saved ONNX model: {saved_models['onnx_path']}")
    copied_models = copy_model_artifacts(saved_models, model_export_dir)
    if copied_models.get("pt_path") is not None:
        print(f"Copied PyTorch model: {copied_models['pt_path']}")
    if copied_models.get("onnx_path") is not None:
        print(f"Copied ONNX model: {copied_models['onnx_path']}")

    print(f"Start prediction for {gate}")
    pred_path = "./Raw_Data_pred"
    process_table(
        x_axis,
        y_axis,
        gate_pre,
        gate,
        pred_path,
        convex=convex,
        seq=(gate_pre is not None),
        dest=dest,
        save_png=save_png,
        force_rebuild=force_rebuild,
    )
    train_test_val_split(gate, pred_path, dest, "pred")

    model_path = saved_models["pt_path"]
    data_df_pred = UNITO_gating(
        model_path,
        x_axis,
        y_axis,
        gate,
        path_raw,
        n_worker,
        device,
        save_prediction_path,
        dest,
        seq=(gate_pre is not None),
        gate_pre=gate_pre,
    )

    accuracy, recall, precision, f1 = evaluation(data_df_pred, gate)
    print(f"{gate}: accuracy:{accuracy}, recall:{recall}, precition:{precision}, f1 score:{f1}")

    plot_all(gate_pre, gate, x_axis, y_axis, path_raw, save_figure_path)
    print("All UNITO prediction visualization saved")

print("Seqential autogating prediction finished")

hyperparameter_df.to_csv("./hyperparameter_tunning.csv", index=False)
