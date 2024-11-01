import torch
import numpy as np
from torch.utils.data import Dataset
from utils import load_arc_json, convert_to_onehot3d, identity_fn, pad_11x32x32

class FlattenTrainDataset(Dataset):
    def __init__(
            self,
            fpath,
            apply_onehot : bool = True,
            apply_padding : bool = True
        ):
        super(FlattenTrainDataset, self).__init__()
        
        arc_dataset = load_arc_json(fpath=fpath)
        inputs = []
        outputs = []
        for task in list(arc_dataset.values()):
            for in_ in task['train']:
                inputs.append(in_['input'])
                outputs.append(in_['output']) 

        self.onehot_fn = convert_to_onehot3d if apply_onehot else identity_fn
        self.pad_fn = pad_11x32x32 if apply_padding else identity_fn
        self.inputs = self.transform(inputs)
        self.outputs = self.transform(outputs)

    def transform(self, data:list):
        data = [self.pad_fn(self.onehot_fn(np.array(datum))) for datum in data]
        data = np.stack(data)
        return data

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]

    def __len__(self):
        return len(self.inputs)

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
        self.onehot_fn = convert_to_onehot3d if apply_onehot else identity_fn
        self.pad_fn = pad_11x32x32 if apply_padding else identity_fn

    def __getitem__(self, index):
        xys_in_task = self.arc_dataset[self.tasks[index]]['train']

        self.xs_in_task = np.stack([
            self.pad_fn(
                self.onehot_fn(
                    np.array(xy['input'])
                )
            )
            for xy in xys_in_task
        ]
        )

        self.ys_in_task = np.stack([
            self.pad_fn(
                self.onehot_fn(
                    np.array(xy['output'])
                )
            )
            for xy in xys_in_task]
        )

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
        self.onehot_fn = convert_to_onehot3d if apply_onehot else identity_fn
        self.pad_fn = pad_11x32x32 if apply_padding else identity_fn
    
    def __getitem__(self, index):
        x_test = self.arc_test[self.tasks[index]]['test'][0]['input']
        y_test = self.arc_solution[self.tasks[index]][0]
        x_test = self.pad_fn(
            self.onehot_fn(
                np.array(x_test)
            )
        )
        y_test = self.pad_fn(
            self.onehot_fn(
                np.array(y_test)
            )
        )
        return x_test, y_test
    
    def __len__(self):
        return len(self.tasks)

class ArcTestDataset(Dataset):
    def __init__(
        self,
        test_fpath,
        apply_onehot:bool = True,
        apply_padding:bool = True
    ):
        self.arc_test = load_arc_json(test_fpath)
        self.tasks = list(self.arc_test.keys())
        self.onehot_fn = convert_to_onehot3d if apply_onehot else identity_fn
        self.pad_fn = pad_11x32x32 if apply_padding else identity_fn

    def __getitem__(self, index):
        x_test = self.arc_test[self.tasks[index]]['test']['input']
        x_test = self.pad_fn(
            self.onehot_fn(
                np.array(x_test)
            )
        )
        return np.array(x_test)
 
    def __len__(self):
        return len(self.tasks)