import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sn


IMAGE_SIZE = 101


def normalize(data, column):
    """
    Normalize one cytometric measurement column to the [0, 1] range.
    """
    values = data[column].to_numpy(dtype=np.float32, copy=True)
    if values.size == 0:
        return values

    min_value = float(values.min())
    max_value = float(values.max())

    if max_value == min_value:
        return np.zeros_like(values, dtype=np.float32)

    return (values - min_value) / (max_value - min_value)


def matrix_plot(data_df_selected, x_axis, y_axis, pad_number=0):
    """
    Convert normalized coordinates into a fixed 101x101 density map.
    """
    density = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    if data_df_selected.empty:
        return density

    x_coord = np.clip(
        np.rint(data_df_selected[x_axis].to_numpy(dtype=np.float32)),
        0,
        IMAGE_SIZE - 1,
    ).astype(np.int16)
    y_coord = np.clip(
        np.rint(data_df_selected[y_axis].to_numpy(dtype=np.float32)),
        0,
        IMAGE_SIZE - 1,
    ).astype(np.int16)

    np.add.at(density, (IMAGE_SIZE - 1 - x_coord, y_coord), 1)

    if pad_number:
        density[density > 0] += pad_number

    return density


def save_density_png(density, output_path):
    """
    Save a density map only when visualization artifacts are requested.
    """
    fig = plt.figure()
    vmax = density.max() / 2 if density.max() > 0 else 1
    vmin = density.min() / 2 if density.min() > 0 else 0
    sn.heatmap(density, vmax=vmax, vmin=vmin)
    plt.savefig(output_path)
    plt.close(fig)


def export_matrix(
    file_name,
    x_axis,
    y_axis,
    gate_pre,
    gate,
    path_raw,
    convex,
    seq=False,
    dest=".",
    save_png=False,
    force_rebuild=False,
):
    """
    Convert one cytometric table into raw and mask numpy matrices.
    """
    raw_numpy_path = os.path.join(f"{dest}/Data/Data_{gate}/Raw_Numpy/", file_name + ".npy")
    mask_numpy_path = os.path.join(f"{dest}/Data/Data_{gate}/Mask_Numpy/", file_name + ".npy")

    if (
        not force_rebuild
        and os.path.exists(raw_numpy_path)
        and os.path.exists(mask_numpy_path)
    ):
        return

    usecols = [x_axis, y_axis, gate]
    if seq and gate_pre is not None:
        usecols.append(gate_pre)

    data_df = pd.read_csv(os.path.join(path_raw, file_name), usecols=usecols)
    if seq and gate_pre is not None:
        data_df = data_df[data_df[gate_pre] == 1]

    data_df_selected = data_df[[x_axis, y_axis, gate]].copy()
    data_df_selected.loc[:, x_axis] = normalize(data_df_selected, x_axis) * (IMAGE_SIZE - 1)
    data_df_selected.loc[:, y_axis] = normalize(data_df_selected, y_axis) * (IMAGE_SIZE - 1)

    raw_density = matrix_plot(data_df_selected, x_axis, y_axis)
    max_value = float(raw_density.max())
    if max_value > 0:
        raw_density = raw_density / max_value
    raw_density = raw_density.astype(np.float32, copy=False)

    data_df_masked = data_df_selected[data_df_selected[gate] == 1]
    mask_density = matrix_plot(data_df_masked, x_axis, y_axis)
    mask_density = (mask_density > 0).astype(np.uint8, copy=False)
    if np.sum(mask_density) > 3:
        mask_density = fill_hull(mask_density, convex)

    np.save(raw_numpy_path, raw_density)
    np.save(mask_numpy_path, mask_density.astype(np.uint8, copy=False))

    if save_png:
        raw_png_path = os.path.join(f"{dest}/Data/Data_{gate}/Raw_PNG/", file_name + ".png")
        mask_png_path = os.path.join(f"{dest}/Data/Data_{gate}/Mask_PNG/", file_name + ".png")
        save_density_png(raw_density, raw_png_path)
        save_density_png(mask_density, mask_png_path)


def process_table(
    x_axis,
    y_axis,
    gate_pre,
    gate,
    directory,
    convex=True,
    seq=False,
    dest=".",
    save_png=False,
    force_rebuild=False,
):
    """
    Prepare image matrices for model training or prediction.
    """
    base_dir = f"{dest}/Data/Data_{gate}"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(f"{base_dir}/Mask_Numpy", exist_ok=True)
    os.makedirs(f"{base_dir}/Raw_Numpy", exist_ok=True)

    if save_png:
        os.makedirs(f"{base_dir}/Mask_PNG", exist_ok=True)
        os.makedirs(f"{base_dir}/Raw_PNG", exist_ok=True)

    name_list = sorted(
        filename
        for filename in os.listdir(directory)
        if filename.endswith(".csv") and os.path.isfile(os.path.join(directory, filename))
    )

    for filename in name_list:
        export_matrix(
            filename,
            x_axis,
            y_axis,
            gate_pre,
            gate,
            directory,
            convex,
            seq=seq,
            dest=dest,
            save_png=save_png,
            force_rebuild=force_rebuild,
        )

    print("process table finished")


def filter(path_list):
    """
    Keep only numpy files, guarding against hidden sync artifacts.
    """
    return [path for path in path_list if path.endswith(".npy")]


def train_test_val_split(gate, path, dest=".", train_pred="train"):
    """
    Prepare subject lists for training or prediction.
    """
    os.makedirs(f"{dest}/Data/Data_{gate}/{train_pred}", exist_ok=True)

    subj_list = sorted(x + ".npy" for x in os.listdir(path) if x.endswith(".csv"))
    imgs_ = [f"{dest}/Data/Data_{gate}/Raw_Numpy/" + x for x in subj_list]
    masks_ = [f"{dest}/Data/Data_{gate}/Mask_Numpy/" + x for x in subj_list]

    imgs = filter(imgs_)
    masks = filter(masks_)
    path = pd.DataFrame(list(zip(imgs, masks)), columns=["Image", "Mask"])

    path.to_csv(f"{dest}/Data/Data_{gate}/{train_pred}/subj.csv", index=False)


def fill_hull(image, convex=True):
    """
    Fill the binary gate region with either a convex hull or contour fill.
    """
    if convex:
        points = np.argwhere(image).astype(np.int16)
        hull = scipy.spatial.ConvexHull(points)
        convex_points = np.array([points[vertex] for vertex in hull.vertices], dtype=np.int64)
        convex_points[:, [1, 0]] = convex_points[:, [0, 1]]

        black_frame = np.zeros(image.shape, dtype=np.uint8)
        cv2.fillPoly(black_frame, pts=[convex_points], color=1)
    else:
        black_frame = image.copy().astype(np.uint8)
        contours, _ = cv2.findContours(
            black_frame,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(black_frame, contours, -1, color=1, thickness=cv2.FILLED)

    return black_frame.astype(np.uint8, copy=False)
