#%%
import os
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from dataloader import ArcTrainDataset, ArcEvaluationDataset, ArcTestDataset, FlattenTrainDataset
from model import OfflineEncoderDecoder, OnlineAttentionSolver, count_parameters

def train_one_epoch(epoch_index, tb_writer, training_loader, optimizer, device):
    running_loss = 0.
    last_loss = 0.

    for idx, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, outputs = data
        inputs = torch.tensor(inputs, device=device)
        outputs = torch.tensor(outputs, device=device)
        n_inputs = inputs.size(0)
        n_outputs = outputs.size(0)
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        input_preds = model(inputs)
        output_preds = model(outputs)

        # Compute the loss and its gradients
        loss = input_latent_loss(input_preds, inputs) / n_inputs \
            + output_latent_loss(output_preds, outputs) / n_outputs

        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if idx % 10 == 9:
            last_loss = running_loss / 10 # loss per batch
            print('  batch {} loss: {}'.format(idx + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + idx + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


arc_train_path = './data/arc-agi_training_challenges.json'
arc_train_sol_path = './data/arc-agi_training_solutions.json'

arc_eval_path = './data/arc-agi_evaluation_challenges.json'
arc_eval_sol_path = './data/arc-agi_evaluation_solutions.json'

arc_test_path = './data/arc-agi_test_challenges.json'

# %%
arc_train_dataset = FlattenTrainDataset(fpath=arc_train_path, apply_onehot=True)
arc_evaluate_dataset = ArcEvaluationDataset(
    test_fpath=arc_train_path,
    solution_fpath=arc_train_sol_path,
    apply_onehot=True
)
# %%
device = torch.device('cpu')
model = OfflineEncoderDecoder(channels=[11,11,11], latent_dim=128)
model.to(device)

# %%
### Test
# x,y = arc_train_dataset[0]
# x_preds = model(torch.tensor(x, dtype=torch.float32))
arc_train_loader = torch.utils.data.DataLoader(arc_train_dataset, batch_size=8)
# %%

# learning for latent dimension
input_latent_loss = torch.nn.MSELoss()
output_latent_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/arc_trainer_{}'.format(timestamp))
epoch_number = 0

EPOCHS = 5

best_vloss = 1_000_000.

for epoch in range(1):
    print('EPOCH {}:'.format(epoch_number + 1))
    model.train()
    avg_loss = train_one_epoch(
        epoch_number,
        writer,
        training_loader=arc_train_loader,
        optimizer=optimizer,
        device=device)
    print(avg_loss)
