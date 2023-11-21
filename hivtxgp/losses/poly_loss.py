import torch
import torch.nn as nn
import numpy as np
from shapely.geometry import Point, Polygon


class PolygonLoss(nn.Module):
    def __init__(self, polygon_coords_list, polygon_loss_weight):
        super(PolygonLoss, self).__init__()
        self.polygon_coords_list = polygon_coords_list
        self.polygon_loss_weight = polygon_loss_weight

    def forward(self, last_point):
        # Calculate the distance to each polygon for each element in the batch
        batch_size = last_point.shape[0]

        # Initialize a list to store distances for each element in the batch
        distances = []

        for i in range(batch_size):
            element_distances = []  # Distances for one element in the batch
            for polygon_coords in self.polygon_coords_list[i]:
                # Convert last_point to a NumPy array on CPU
                last_point_cpu = last_point[i].cpu().detach().numpy()

                element_point = []  # Store individual Points for this element
                for coord in last_point_cpu:
                    point = Point(coord[0], coord[1])
                    element_point.append(point)

                # Create a Shapely Polygon from the polygon_coords
                polygon = Polygon(polygon_coords)

                # Calculate the minimum distance between the Points and the polygon
                min_distance = min(point.distance(polygon) for point in element_point)
                element_distances.append(min_distance)

            distances.append(element_distances)

        # Find the minimum distance for each element in the batch
        for i in range(batch_size):
            if distances[i] == []:
                min_distances = 0
            else:
                min_distances = min(distances[i])

        # Define a loss term based on the minimum distance
        polygon_loss = self.polygon_loss_weight * torch.tensor(min_distances, device=last_point.device)

        return polygon_loss