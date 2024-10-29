import torch
import numpy as np
from torch.utils.data import Dataset
from utils import load_arc_json, convert_to_onehot3d

# As Arc Data is small enough
# I put all the data allocated into the dataset.
class ArcTrainDataset(Dataset):
    def __init__(
            self,
            fpath,
            apply_onehot:bool=True,
            apply_padding:bool=True
        ):
        super(ArcTrainDataset, self).__init__()

        self.arc_dataset = load_arc_json(fpath=fpath)
        self.tasks = list(self.arc_dataset.keys())
        self.apply_onehot = apply_onehot

    def __getitem__(self, index):
        xys_in_task = self.arc_dataset[self.tasks[index]]['train']
        if self.apply_onehot:
            self.xs_in_task = [
                convert_to_onehot3d(
                    np.array(xy['input'])
                )
                for xy in xys_in_task
            ]
            print(self.xs_in_task)
            self.ys_in_task = [
                convert_to_onehot3d(
                    np.array(xy['output'])
                )
                for xy in xys_in_task
            ]
        else:
            self.xs_in_task = [
                np.array(xy['input'])
                for xy in xys_in_task
            ]
            self.ys_in_task = [
                np.array(xy['output'])
                for xy in xys_in_task
            ]
        return self.xs_in_task, self.ys_in_task

    def __len__(self):
        return len(self.tasks)

class ArcEvaluationDataset(Dataset):
    def __init__(
        self,
        test_fpath,
        solution_fpath,
        apply_onehot: bool = True,
        apply_padding:bool=True
    ):
        self.arc_test = load_arc_json(test_fpath)
        self.arc_solution = load_arc_json(solution_fpath)
        self.tasks = list(self.arc_test.keys())
        self.apply_onehot = apply_onehot
    
    def __getitem__(self, index):
        x_test = self.arc_test[self.tasks[index]]['test'][0]['input']
        y_test = self.arc_solution[self.tasks[index]][0]
        if self.apply_onehot:
            x_test = convert_to_onehot3d(
                np.array(x_test)
            )
            y_test = convert_to_onehot3d(
                np.array(y_test)
            )
        else:
            x_test = np.array(x_test)
            y_test = np.array(y_test)
        return x_test, y_test
    
    def __len__(self):
        return len(self.tasks)

class ArcTestDataset(Dataset):
    def __init__(
        self,
        test_fpath,
        apply_onehot:bool = True
    ):
        self.arc_test = load_arc_json(test_fpath)
        self.tasks = list(self.arc_test.keys())
        self.apply_onehot = apply_onehot
    def __getitem__(self, index):
        x_test = self.arc_test[self.tasks[index]]['test']['input']
        if self.apply_onehot:
            return convert_to_onehot3d(
                np.array(x_test)
            )
        else:
            return np.array(x_test)
    def __len__(self):
        return len(self.tasks)

def pad_10x32x32(grid, pad_value=0, dtype=torch.float32):
    padded_grid = torch.full((10, 32, 32), pad_value, dtype=dtype)
    channel, height, width = len(grid), len(grid[0]), len(grid[0][0])
    padded_grid[:,:height, :width] = torch.tensor(grid, dtype=dtype)
    return padded_grid