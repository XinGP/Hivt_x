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
import torch
import torch.nn as nn

class FinalPredictionL1Loss(nn.Module):

    def __init__(self, reduction: str = 'mean') -> None:
        super(FinalPredictionL1Loss, self).__init__()
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        # Assuming pred and target have shape (batch_size, num_time_steps, num_features)
        # Calculate the L1 loss only for the final time step
        final_pred = pred[:, :]  # Select the predictions at the final time step
        l1_loss = torch.abs(final_pred - target).sum(dim=-1)  # L1 loss for the final time step

        if self.reduction == 'mean':
            return l1_loss.mean()
        elif self.reduction == 'sum':
            return l1_loss.sum()
        elif self.reduction == 'none':
            return l1_loss
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))
