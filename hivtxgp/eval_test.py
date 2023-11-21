# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch_geometric.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import ArgoverseV1Dataset
from models.hivt import HiVT
from models_future.hivt import HiVT_future
import torch.nn.functional as F

from losses import LaplaceNLLLoss
from losses import SoftTargetCrossEntropyLoss
from metrics import ADE
from metrics import FDE
from metrics import MR

torch.autograd.set_detect_anomaly(True)
if __name__ == '__main__':
    pl.seed_everything(2022)

    # y_one
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--gpus', type=int, default=1)
    args = parser.parse_args()

    # future
    parser_future = ArgumentParser()
    parser_future.add_argument('--train_batch_size', type=int, default=8)
    parser_future.add_argument('--val_batch_size', type=int, default=8)
    parser_future.add_argument('--shuffle', type=bool, default=True)
    parser_future.add_argument('--num_workers', type=int, default=4)
    parser_future.add_argument('--pin_memory', type=bool, default=True)
    parser_future.add_argument('--persistent_workers', type=bool, default=True)
    parser_future.add_argument('--gpus', type=int, default=1)
    parser_future.add_argument('--max_epochs', type=int, default=1)
    parser_future.add_argument('--monitor', type=str, default='val_minFDE', choices=['val_minADE', 'val_minFDE', 'val_minMR'])
    parser_future.add_argument('--save_top_k', type=int, default=5)
    parser_future.add_argument('--embed_dim', type=int, default=64)
    parser_future = HiVT_future.add_model_specific_args(parser_future)
    args_future = parser_future.parse_args()

    model_checkpoint = ModelCheckpoint(monitor=args_future.monitor, save_top_k=args_future.save_top_k, mode='min')
    trainer_future = pl.Trainer.from_argparse_args(args_future, callbacks=[model_checkpoint])
    model_future = HiVT_future(**vars(args_future))

    trainer = pl.Trainer.from_argparse_args(args)
    model = HiVT.load_from_checkpoint(checkpoint_path='/home/xin/HiVT/checkpoints/HiVT-128/checkpoints/epoch=63-step=411903.ckpt', parallel=True)
    val_dataset = ArgoverseV1Dataset(root='/home/xin/下载/argodata', split='val', local_radius=model.hparams.local_radius)
    dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=4,
                            pin_memory=True, persistent_workers=True)

    model.eval()
    model_future.train()
    optimizer_future = torch.optim.Adam(model_future.parameters(), lr=5e-4, weight_decay=1e-4)
    avg_loss = 0

    device = torch.device("cuda")
    model.to(device)
    model_future.to(device)
    ade_metric = ADE().to(device)
    fde_metric = FDE().to(device)
    mr_metric = MR().to(device)
    for i, batch in enumerate(dataloader):

        inputs = batch.to(device)
        y_hat_128, pi_128 = model(inputs)
        """positions = batch['positions'].to(device)
        position_xy = torch.zeros(positions.size(0), 50, 2, device=device)
        position_y = y_one[:, :, :2] + positions[:, 19].unsqueeze(-2)
        position_xy[:, :20, :] = positions[:, :20, :]
        position_xy[:, 20:, :] = position_y[:, :, :]
        xy = torch.cat((batch['x'], y_one[:, :, :2]), dim=1)

        batch.position_xys = position_xy
        batch.xys = xy

        inputs_future = batch.to(device)
        y_hat, pi = model_future(inputs_future)"""

        reg_mask = ~batch['padding_mask'][:, 20:]
        l2_norm = (torch.norm(y_hat_128[:, :, :, : 2] - batch.y, p=2, dim=-1) * reg_mask).sum(dim=-1)  # [F, N]
        best_mode = l2_norm.argmin(dim=0)
        y_hat_best = y_hat_128[best_mode, torch.arange(batch.num_nodes)]
        reg_loss = LaplaceNLLLoss(reduction='mean')(y_hat_best[reg_mask], batch.y[reg_mask])
        print('val_reg_loss', reg_loss)

        y_hat_agent = y_hat_128[:, batch['agent_index'], :, : 2]
        y_agent = batch.y[batch['agent_index']]
        fde_agent = torch.norm(y_hat_agent[:, :, -1] - y_agent[:, -1], p=2, dim=-1)
        best_mode_agent = fde_agent.argmin(dim=0)

        y_hat_best_agent = y_hat_agent[best_mode_agent, torch.arange(batch.num_graphs)]

        ade_metric.update(y_hat_best_agent, y_agent)
        fde_metric.update(y_hat_best_agent, y_agent)
        mr_metric.update(y_hat_best_agent, y_agent)

        print("ADE", ade_metric.compute())
        print("FDE", fde_metric.compute())
        print("MR", mr_metric.compute())

    #torch.save(model_future, "hivt_future.pth")



