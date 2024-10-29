#%%
import os

import torch
import numpy as np
from dataloader import ArcTrainDataset, ArcEvaluationDataset, ArcTestDataset
from model import UpEncoder

arc_train_path = './data/arc-agi_training_challenges.json'

arc_train_sol_path = './data/arc-agi_training_solutions.json'

arc_eval_path = './data/arc-agi_evaluation_challenges.json'
arc_eval_sol_path = './data/arc-agi_evaluation_solutions.json'

arc_test_path = './data/arc-agi_test_challenges.json'


#
# %%
arc_train_dataset = ArcTrainDataset(fpath=arc_train_path, apply_onehot=True)
arc_evaluate_dataset = ArcEvaluationDataset(
    test_fpath=arc_train_path,
    solution_fpath=arc_train_sol_path,
    apply_onehot=True
)
# %%
model = UpEncoder()
# %%
### Test
x,y = arc_train_dataset[0]
model(torch.tensor(x, dtype=torch.float32))