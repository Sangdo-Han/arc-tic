import os
import json
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from typing import Optional

def identity_fn(x):
    return x

def load_arc_json(fpath:str) -> dict:
    with open(fpath, 'r') as jsf:
        arc_dict = json.load(jsf)
    return arc_dict

# C x H x W (for torch shape, 10 x H x W)
# input : 2d np.array with digit of colors
def convert_to_onehot3d(arr:np.array, level:int=11) -> np.array:
    arr = arr + 1
    out = np.zeros( (level, arr.size))
    out[arr.ravel(), np.arange(arr.size)] = 1
    out.shape = (level, ) + arr.shape
    return out

def pad_11x32x32(grid_10xhxw, pad_value=0, dtype=np.float32):
    padded_grid = np.full((11,32,32), fill_value=pad_value, dtype=dtype)
    channel, height, width = grid_10xhxw.shape
    padded_grid[:, :height, :width] = grid_10xhxw
    return padded_grid

# return to 2d np.array with digits of colors
def convert_onehot_to_origin(arr_3d:np.array, level:int=11) -> np.array:
    pjt_arr = np.arange(level)
    pjt_arr = pjt_arr[:, np.newaxis, np.newaxis]
    arr_3d = arr_3d * pjt_arr # element-wise production
    out = arr_3d.sum(axis=0)
    out = out -1
    return out

def convert_to_xyz_points(arr:np.array) -> tuple:
    hex_colors = ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
    y_len, x_len = arr.shape
    listed = arr.flatten()
    cs = []
    xs = []
    ys = []
    zs = []
    for idx, color_idx in enumerate(listed):
        cs.append(hex_colors[color_idx])
        xs.append(idx % x_len)
        ys.append(idx // x_len)
        zs.append(int(color_idx))
    return xs, ys, zs, cs

def vis_train_grid(
        train_sample:list,
        save_as : Optional[str] = 'sample.png'
) -> None:

    hex_colors = ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
    cmap = colors.ListedColormap(
        hex_colors
    )
    norm = colors.Normalize(vmin=0, vmax=9)

    n_samples = len(train_sample)
    fig, axes = plt.subplots(nrows=2, ncols=n_samples)

    for sample_idx in range(n_samples):
        input_arr = train_sample[sample_idx]['input']
        output_arr = train_sample[sample_idx]['output']
        axes[0][sample_idx].imshow(input_arr, cmap=cmap, norm=norm)
        axes[1][sample_idx].imshow(output_arr, cmap=cmap, norm=norm)
    if save_as:
        fig.savefig(save_as)

def vis_train_in3d(
        train_sample:list,
        save_as : Optional[str] = 'pointcloud_sample.png',
        apply_colors : bool = True
) -> None:
    n_samples = len(train_sample)
    fig, axes = plt.subplots(
        nrows=2, ncols=n_samples, subplot_kw={"projection":"3d"}
    )

    for sample_idx in range(n_samples):
        xs_in,ys_in,zs_in,cs_in = convert_to_xyz_points(
            np.array(train_sample[sample_idx]['input'])
        )
        xs_out,ys_out,zs_out,cs_out = convert_to_xyz_points(
            np.array(train_sample[sample_idx]['output'])
        )

        if not apply_colors:
            cs_in = None
            cs_out = None

        axes[0][sample_idx].scatter(xs_in,ys_in,zs_in,c=cs_in)
        axes[1][sample_idx].scatter(xs_out,ys_out,zs_out,c=cs_out)
    
    if save_as:
        fig.savefig(save_as)

def visualize_all_trainset(
        train_dataset,
        rootdir : str = './data/visualization',
        apply_colors : bool = False
) -> None:
    os.makedirs(rootdir, exist_ok=True)

    train_sample_keys = list(train_dataset.keys())
    
    for train_sample_key in train_sample_keys:
        train_sample = train_dataset[train_sample_key]['train']
        vis_train_grid(
            train_sample,
            save_as=os.path.join(rootdir, f'{train_sample_key}.png')
        )
        vis_train_in3d(
            train_sample,
            save_as=os.path.join(rootdir, f'{train_sample_key}_pointcloud.png'),
            apply_colors=apply_colors
        )
