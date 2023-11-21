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
import os
from itertools import permutations
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from argoverse.map_representation.map_api import ArgoverseMap
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from tqdm import tqdm

from utils import TemporalData

# TNT___________________________________________________________
import math
import argparse
from os.path import join as pjoin
import sys

from matplotlib.collections import LineCollection
from matplotlib import pyplot as plt
import copy
import warnings
#from torch.utils.data import Dataset, DataLoader
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.visualization.visualize_sequences import viz_sequence
from argoverse.utils.mpl_plotting_utils import visualize_centerline
from argoverse.utils.centerline_utils import centerline_to_polygon

from PythonRobotics_1.PathPlanning.CubicSpline import cubic_spline_planner
from core.util.preprocessor.base import frenet_optimal_planning
from core.util.cubic_spline import Spline2D
warnings.filterwarnings("ignore")
RESCALE_LENGTH = np.float32(1.0)    # the rescale length th turn the lane vector into equal distance pieces

def dist(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
# TNT___________________________________________________________

class ArgoverseV1Dataset(Dataset):

    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 local_radius: float = 50,
                 start_index: int = 0) -> None:
        self._split = split
        self._local_radius = local_radius
        self._url = f'https://s3.amazonaws.com/argoai-argoverse/forecasting_{split}_v1.1.tar.gz'
        if split == 'sample':
            self._directory = 'forecasting_sample'
        elif split == 'train':
            self._directory = 'train'
        elif split == 'val':
            self._directory = 'val'
        elif split == 'test':
            self._directory = 'test_obs'
        else:
            raise ValueError(split + ' is not valid')
        self.root = root
        self._start_index = start_index
        self._raw_file_names = os.listdir(self.raw_dir)
        self._processed_file_names = [os.path.splitext(f)[0] + '.pt' for f in self.raw_file_names]
        self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]
        super(ArgoverseV1Dataset, self).__init__(root, transform=transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'data')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self._directory, 'processed')

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    @property
    def processed_paths(self) -> List[str]:
        return self._processed_paths

    def process(self) -> None:
        am = ArgoverseMap()
        #for idx in range(self._start_index, len(self.raw_paths)):
        for raw_path in tqdm(self.raw_paths):
            #raw_path = self.raw_paths[idx]
            kwargs = process_argoverse(self._split, raw_path, am, self._local_radius)
            data = TemporalData(**kwargs)
            torch.save(data, os.path.join(self.processed_dir, str(kwargs['seq_id']) + '.pt'))

    def len(self) -> int:
        return len(self._raw_file_names)

    def get(self, idx) -> Data:
        return torch.load(self.processed_paths[idx])


def process_argoverse(split: str,
                      raw_path: str,
                      am: ArgoverseMap,
                      radius: float) -> Dict:
    df = pd.read_csv(raw_path)
    city = df['CITY_NAME'].values[0]

    # xin_____________________________
    timestamps = list(np.sort(df['TIMESTAMP'].unique()))
    historical_timestamps = timestamps[: 20]
    historical_df = df[df['TIMESTAMP'].isin(historical_timestamps)]
    actor_ids = list(historical_df['TRACK_ID'].unique())
    df = df[df['TRACK_ID'].isin(actor_ids)]
    num_nodes = len(actor_ids)

    av_df = df[df['OBJECT_TYPE'] == 'AV'].iloc
    av_index = actor_ids.index(av_df[0]['TRACK_ID'])
    agent_df = df[df['OBJECT_TYPE'] == 'AGENT'].iloc
    agent_index = actor_ids.index(agent_df[0]['TRACK_ID'])

    # make the scene centered at AV
    origin = torch.tensor([av_df[19]['X'], av_df[19]['Y']], dtype=torch.float)
    av_heading_vector = origin - torch.tensor([av_df[18]['X'], av_df[18]['Y']], dtype=torch.float)
    theta = torch.atan2(av_heading_vector[1], av_heading_vector[0])
    rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                               [torch.sin(theta), torch.cos(theta)]])

    # initialization
    x = torch.zeros(num_nodes, 50, 2, dtype=torch.float)
    edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
    padding_mask = torch.ones(num_nodes, 50, dtype=torch.bool)
    bos_mask = torch.zeros(num_nodes, 20, dtype=torch.bool)
    rotate_angles = torch.zeros(num_nodes, dtype=torch.float)
    agent_traj = torch.zeros(50, 2, dtype=torch.float)

    for actor_id, actor_df in df.groupby('TRACK_ID'):
        node_idx = actor_ids.index(actor_id)
        node_steps = [timestamps.index(timestamp) for timestamp in actor_df['TIMESTAMP']]
        padding_mask[node_idx, node_steps] = False
        if padding_mask[node_idx, 19]:  # make no predictions for actors that are unseen at the current time step
            padding_mask[node_idx, 20:] = True
        xy = torch.from_numpy(np.stack([actor_df['X'].values, actor_df['Y'].values], axis=-1)).float()
        if node_idx == agent_index:
            agent_traj = xy
        x[node_idx, node_steps] = torch.matmul(xy - origin, rotate_mat)
        node_historical_steps = list(filter(lambda node_step: node_step < 20, node_steps))
        if len(node_historical_steps) > 1:  # calculate the heading of the actor (approximately)
            heading_vector = x[node_idx, node_historical_steps[-1]] - x[node_idx, node_historical_steps[-2]]
            rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
        else:  # make no predictions for the actor if the number of valid time steps is less than 2
            padding_mask[node_idx, 20:] = True

    # bos_mask is True if time step t is valid and time step t-1 is invalid
    bos_mask[:, 0] = ~padding_mask[:, 0]
    bos_mask[:, 1: 20] = padding_mask[:, : 19] & ~padding_mask[:, 1: 20]

    positions = x.clone()

    x[:, 20:] = torch.where((padding_mask[:, 19].unsqueeze(-1) | padding_mask[:, 20:]).unsqueeze(-1),
                            torch.zeros(num_nodes, 30, 2),
                            x[:, 20:] - x[:, 19].unsqueeze(-2))
    x[:, 1: 20] = torch.where((padding_mask[:, : 19] | padding_mask[:, 1: 20]).unsqueeze(-1),
                              torch.zeros(num_nodes, 19, 2),
                              x[:, 1: 20] - x[:, : 19])
    x[:, 0] = torch.zeros(num_nodes, 2)

    # X---------------------------------------------------

    v1 = torch.sqrt(
        torch.pow((agent_traj[19][0] - agent_traj[18][0]), 2) + torch.pow((agent_traj[19][1] - agent_traj[18][1]),
                                                                          2)) * 10
    v2 = torch.sqrt(
        torch.pow((agent_traj[19][0] - agent_traj[17][0]), 2) + torch.pow((agent_traj[19][1] - agent_traj[17][1]),
                                                                          2)) * 5
    v = (v1 + v2) / 2
    ctr_line_candts = am.get_candidate_centerlines_for_traj(agent_traj[:20, :], city, viz=False)
    agt_traj_obs_matmul = torch.matmul(agent_traj[:20, :] - origin, rotate_mat)
    ctr_line_list = []
    for ctr_line in ctr_line_candts:
        ctr_line = torch.as_tensor(ctr_line, dtype=torch.float)
        result = torch.matmul(ctr_line - origin, rotate_mat)
        ctr_line_list.append(result)
    # 裁剪中心线
    max_xy = get_ref_centerline_And_future(ctr_line_list, agt_traj_obs_matmul.numpy(), v.numpy())
    # X---------------------------------------------------

    # get lane features at the current time step
    df_19 = df[df['TIMESTAMP'] == timestamps[19]]
    node_inds_19 = [actor_ids.index(actor_id) for actor_id in df_19['TRACK_ID']]
    node_positions_19 = torch.from_numpy(np.stack([df_19['X'].values, df_19['Y'].values], axis=-1)).float()

    (lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index,
     lane_actor_vectors, lane_actor_index_max, lane_actor_vectors_max) = get_lane_features(am, node_inds_19, node_positions_19, origin, rotate_mat, city, radius, max_xy)

    y = None if split == 'test' else x[:, 20:]
    seq_id = os.path.splitext(os.path.basename(raw_path))[0]

    return {
        'x': x[:, : 20],  # [N, 20, 2]
        'positions': positions,  # [N, 50, 2]
        'edge_index': edge_index,  # [2, N x N - 1]
        'y': y,  # [N, 30, 2]
        'num_nodes': num_nodes,
        'padding_mask': padding_mask,  # [N, 50]
        'bos_mask': bos_mask,  # [N, 20]
        'rotate_angles': rotate_angles,  # [N]
        'lane_vectors': lane_vectors,  # [L, 2]
        'is_intersections': is_intersections,  # [L]
        'turn_directions': turn_directions,  # [L]
        'traffic_controls': traffic_controls,  # [L]
        'lane_actor_index': lane_actor_index,  # [2, E_{A-L}]
        'lane_actor_vectors': lane_actor_vectors,  # [E_{A-L}, 2]
        'lane_actor_index_max': lane_actor_index_max,  # [2, E_{A-L}]
        'lane_actor_vectors_max': lane_actor_vectors_max,  # [E_{A-L}, 2]
        'seq_id': int(seq_id),
        'av_index': av_index,
        'agent_index': agent_index,
        'city': city,
        'origin': origin.unsqueeze(0),
        'theta': theta,
    }


def get_lane_features(am: ArgoverseMap,
                      node_inds: List[int],
                      node_positions: torch.Tensor,
                      origin: torch.Tensor,
                      rotate_mat: torch.Tensor,
                      city: str,
                      radius: float,
                      max_xy: List) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                              torch.Tensor, torch.Tensor, torch.Tensor]:
    lane_positions, lane_vectors, is_intersections, turn_directions, traffic_controls = [], [], [], [], []
    lane_ids = set()
    for node_position in node_positions:
        lane_ids.update(am.get_lane_ids_in_xy_bbox(node_position[0], node_position[1], city, radius))
    node_positions = torch.matmul(node_positions - origin, rotate_mat).float()
    for lane_id in lane_ids:
        lane_centerline = torch.from_numpy(am.get_lane_segment_centerline(lane_id, city)[:, : 2]).float()
        lane_centerline = torch.matmul(lane_centerline - origin, rotate_mat)
        is_intersection = am.lane_is_in_intersection(lane_id, city)
        turn_direction = am.get_lane_turn_direction(lane_id, city)
        traffic_control = am.lane_has_traffic_control_measure(lane_id, city)
        lane_positions.append(lane_centerline[:-1])
        lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1])
        count = len(lane_centerline) - 1
        is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))
        if turn_direction == 'NONE':
            turn_direction = 0
        elif turn_direction == 'LEFT':
            turn_direction = 1
        elif turn_direction == 'RIGHT':
            turn_direction = 2
        else:
            raise ValueError('turn direction is not valid')
        turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
        traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))
    lane_positions = torch.cat(lane_positions, dim=0)
    lane_vectors = torch.cat(lane_vectors, dim=0)
    is_intersections = torch.cat(is_intersections, dim=0)
    turn_directions = torch.cat(turn_directions, dim=0)
    traffic_controls = torch.cat(traffic_controls, dim=0)

    lane_actor_index_init = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous()
    lane_actor_vectors_init = \
        lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1)
    mask = torch.norm(lane_actor_vectors_init, p=2, dim=-1) < radius
    lane_actor_index = lane_actor_index_init[:, mask]
    lane_actor_vectors = lane_actor_vectors_init[mask]

    combined_mask = torch.zeros(lane_actor_vectors_init.size(0), dtype=torch.bool)
    for arr in max_xy:
        for point in arr:
            mask_max = torch.norm(lane_actor_vectors_init - point, p=2, dim=-1) < 20
            combined_mask = combined_mask.logical_or(mask_max)

    lane_actor_index_max = lane_actor_index_init[:, combined_mask]
    lane_actor_vectors_max = lane_actor_vectors_init[combined_mask]

    return lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index, lane_actor_vectors, lane_actor_index_max, lane_actor_vectors_max


def get_ref_centerline_And_future(cline_list, obs, v, **kwargs):
    ref_centerlines = [Spline2D(x=cline_list[i][:, 0], y=cline_list[i][:, 1]) for i in range(len(cline_list))]

    # search the closest point of the traj final position to each center line
    max_xy = []
    for i, line in enumerate(ref_centerlines):
        # xin
        csp = cubic_spline_planner.Spline2D(line.x_fine, line.y_fine)
        s0, d0, _, _ = Spline2D.calc_frenet_position(line, obs[-1][0], obs[-1][1])
        # initial state
        c_d_d = 0.0  # current lateral speed [m/s]
        c_d_dd = 0.0  # current lateral acceleration [m/s]
        fplist = frenet_optimal_planning(
            csp, s0, v, d0, c_d_d, c_d_dd)
        end_s0 = []
        fp_x0 = []
        fp_y0 = []
        for fp in fplist:
            if fp.x:
                end_s = fp.s[-1]
                fp_x = fp.x
                fp_y = fp.y
                # end_point_y = fp.y[-1]
                # end_s, end_d, x_n, y_n = Spline2D.calc_frenet_position(line, end_point_x, end_point_y)
                # 在这里修剪centerline
                end_s0.append(end_s)
                fp_x0.append(fp_x)
                fp_y0.append(fp_y)
            else:
                pass
        if end_s0 == []:
            pass
        else:
            max_s = max(end_s0)
            max_index = end_s0.index(max_s)
            fp_x_max = fp_x0[max_index]
            fp_y_max = fp_y0[max_index]
            max_xy.append(np.stack([fp_x_max, fp_y_max], axis=-1))

    """start_end_points = set()
    unique_tensors = []
    for tensor in max_xy:
        start_point = tuple(tensor[0])
        end_point = tuple(tensor[-1])

        if (start_point, end_point) not in start_end_points:
            unique_tensors.append(tensor)
            start_end_points.add((start_point, end_point))

    max_xy_tensor_filtered = unique_tensors"""

    return max_xy

"""def get_obs_sd(cline_list, pred_gt, traj_obs, **kwargs):
    global filtered_point, traj_d
    if len(cline_list) == 1:
        return [Spline2D(x=cline_list[0][:, 0], y=cline_list[0][:, 1])], 0
    else:
    line_idx = 0
    ref_centerlines = [Spline2D(x=cline_list[i][:, 0], y=cline_list[i][:, 1]) for i in range(len(cline_list))]
    # search the closest point of the traj final position to each center line
    min_distances = []

    for line in ref_centerlines:
        xy = np.stack([line.x_fine, line.y_fine], axis=1)
        diff = xy - pred_gt[-1, :2]
        dis = np.hypot(diff[:, 0], diff[:, 1])
        min_distances.append(np.min(dis))
    line_idx = np.argmin(min_distances)

    all_s = []
    all_d = []
    for j, _ in enumerate(traj_obs):
        os, od, _, _, _ = Spline2D.calc_frenet_position(ref_centerlines[line_idx], traj_obs[j][0], traj_obs[j][1])
        all_s.append(os)
        all_d.append(od)
    return ref_centerlines, line_idx, all_s, all_d"""