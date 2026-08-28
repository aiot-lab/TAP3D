import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch3d.loss import chamfer_distance



def outlier_removal_bound(arr):
        """
        Find the lower and upper bound for the outlier removal
        """
        if len(arr) == 0:
            print("Warning: Empty array passed to percentile")
        # Step 1: Calculate the first quartile (Q1) and third quartile (Q3)
        Q1 = np.percentile(arr, 25)
        Q3 = np.percentile(arr, 75)
        # print('Q1:', Q1, 'Q3:', Q3)
        # Step 2: Calculate the interquartile range (IQR)
        IQR = Q3 - Q1
        # print('IQR:', IQR)
        # Step 3: Define the lower and upper bounds to identify outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # print('lower_bound:', lower_bound, 'upper_bound:', upper_bound)
        # Step 4: Filter out the outliers
        # filtered_arr = arr[(arr >= lower_bound) & (arr <= upper_bound)]
        return lower_bound, upper_bound 

def depth_map_to_point_cloud(depth_map, hfov_deg, vfov_deg):
    """
    Convert a depth map to a 3D point cloud.

    Parameters:
    depth_map: 2D numpy array, the depth map (in meters or any consistent unit).
    hfov_deg: float, horizontal field of view of the camera in degrees.
    vfov_deg: float, vertical field of view of the camera in degrees.

    Returns:
    point_cloud: 3D numpy array of shape (H, W, 3), where (H, W) is the depth map shape.
                Each point has (x, y, z) coordinates in 3D space.
    """
    # Get depth map dimensions
    height, width = depth_map.shape

    # Convert HFoV and VFoV from degrees to radians
    hfov_rad = np.deg2rad(hfov_deg)
    vfov_rad = np.deg2rad(vfov_deg)

    # Calculate focal lengths (fx, fy) based on FoV and image dimensions
    fx = width / (2 * np.tan(hfov_rad / 2))
    fy = height / (2 * np.tan(vfov_rad / 2))
    # Create a grid of pixel coordinates
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    # Normalize pixel coordinates to camera coordinates
    x_cam = (x - width / 2) / fx
    y_cam = (y - height / 2) / fy
    # Calculate 3D coordinates
    z_cam = depth_map
    x_3d = x_cam * z_cam
    y_3d = y_cam * z_cam

    # Stack the coordinates to form the point cloud
    point_cloud = np.stack((x_3d, y_3d, z_cam), axis=-1)

    return point_cloud


class DepthMask2PointCloud(nn.Module):
    '''
    converting the depth_mask_3C to the point_cloud_3C
    '''
    def __init__(self, exp_config):
        super(DepthMask2PointCloud, self).__init__()
        self.sensor_name = exp_config['sensor_name']
        self.max_num_points = exp_config['max_num_points']
        self.max_num_persons = exp_config['max_num_persons']
        if self.sensor_name == 'senxor_m08':
            self.hfov_deg = 90
            self.vfov_deg = 67
            self.height = 62
            self.width = 80
        elif self.sensor_name == 'senxor_m16':
            self.hfov_deg = 88.5 #used to be 88
            self.vfov_deg = 58.5 #used to be 66
            self.height = 120
            self.width = 160
        elif self.sensor_name == 'seek_thermal':
            self.hfov_deg = 81
            self.vfov_deg = 59
            self.height = 150
            self.width = 200
        else:
            raise ValueError(f"Sensor name {self.sensor_name} not supported")
        
    def forward(self, depth_mask_3C):
        '''
        Args:
            depth_mask_3C: torch.Tensor of shape (B, 3, H, W)
        Returns:
            point_cloud_3C: Tensor of shape [B, 3, (max_num_points+1)*max_num_persons]
        '''
        batch_size = depth_mask_3C.shape[0]
        # Initialize output tensor
        point_cloud_3C = torch.zeros(batch_size, 3, (self.max_num_points+1)*self.max_num_persons, 
                                   dtype=depth_mask_3C.dtype, device=depth_mask_3C.device)
        
        for b in range(batch_size):
            depth_map = depth_mask_3C[b, 0].detach().cpu().numpy()  # Depth values
            indicator_map = depth_mask_3C[b, 1].detach().cpu().numpy()  # Person indicators (1, 2, 3, etc.)
            indicator_map = np.round(indicator_map)
            
            # Process each person separately
            for person_idx in range(1, self.max_num_persons + 1):
                # Skip if this person index doesn't exist in the indicator map
                if person_idx not in indicator_map:
                    continue
                
                # Extract this person's depth data
                person_mask = (indicator_map == person_idx)
                person_depth = depth_map.copy()
                person_depth[~person_mask] = 0
                # Convert to point cloud (if there are any points)
                if np.any(person_mask):
                    point_cloud = depth_map_to_point_cloud(person_depth, self.hfov_deg, self.vfov_deg)  # shape: (H,W 3 (x,y,z))
                    # Extract valid points (z > 3), i.e., depth > 0
                    valid_points = point_cloud[person_depth > 3]
                    # Sample or pad the point cloud list
                    if valid_points.shape[0] > 0:
                        # remove outliers
                        person_non_zero = depth_map[depth_map>3]   # unit: mm
                        if person_non_zero.shape[0]==0:
                            continue
                        lower_bound, upper_bound = outlier_removal_bound(person_non_zero.flatten())
                        point_cloud_list = []
                        for i in range(point_cloud.shape[0]):
                            for j in range(point_cloud.shape[1]):
                                if point_cloud[i, j, 2]>3 and point_cloud[i, j, 2]>lower_bound and point_cloud[i, j, 2] < upper_bound:
                                    point_cloud_list.append(point_cloud[i, j]) #PROBLEM: how to process empty case?
                        outlier_removed_points = np.array(point_cloud_list)
                        if (outlier_removed_points.shape[0]==0):
                            continue                        
                        # sample point cloud back to the required number of points
                        points = self._point_cloud_sampling(outlier_removed_points)
                        # Calculate offset in the output tensor for this person
                        offset = (person_idx - 1) * (self.max_num_points + 1)
                        # Copy points to output tensor
                        point_cloud_3C[b, :, offset:offset + self.max_num_points] = torch.tensor(
                            points, dtype=depth_mask_3C.dtype, device=depth_mask_3C.device)
                        # Add indicator point (1,0,0) after the points
                        point_cloud_3C[b, 0, offset + self.max_num_points] = 1.0
        return point_cloud_3C
    
    def _point_cloud_sampling(self, point_cloud):
        """
        Sample or pad the point cloud to have exactly max_num_points
        
        Args:
            point_cloud: numpy array of shape (N, 3) containing the points
        
        Returns:
            Sampled/padded point cloud of shape (3, max_num_points)
        """
        
        num_points = point_cloud.shape[0]
        
        # If the number of points is less than max_num_points, pad the point cloud
        # If more, sample the point cloud
        if num_points < self.max_num_points:
            pad_num = self.max_num_points - num_points
            pad = np.zeros((pad_num, 3))
            point_cloud = np.concatenate([point_cloud, pad], axis=0)  # shape: [max_num_points, 3]
        else:
            # Sample points randomly without replacement
            idx = np.random.choice(num_points, self.max_num_points, replace=False)
            point_cloud = point_cloud[idx]
            
        # Return with shape [3, max_num_points]
        return point_cloud.T


class DepthMask2PointCloudFast(nn.Module):
    '''
    Optimized version of DepthMask2PointCloud with faster processing
    '''
    def __init__(self, exp_config):
        super(DepthMask2PointCloudFast, self).__init__()
        self.sensor_name = exp_config['sensor_name']
        self.max_num_points = exp_config['max_num_points']
        self.max_num_persons = exp_config['max_num_persons']
        
        # Pre-compute camera parameters
        if self.sensor_name == 'senxor_m08':
            self.hfov_deg = 90
            self.vfov_deg = 67
            self.height = 62
            self.width = 80
        elif self.sensor_name == 'senxor_m16':
            self.hfov_deg = 88.5
            self.vfov_deg = 58.5
            self.height = 120
            self.width = 160
        elif self.sensor_name == 'seek_thermal':
            self.hfov_deg = 81
            self.vfov_deg = 59
            self.height = 150
            self.width = 200
        else:
            raise ValueError(f"Sensor name {self.sensor_name} not supported")
        
        # Pre-compute camera intrinsics
        self._setup_camera_intrinsics()
        
    def _setup_camera_intrinsics(self):
        """Pre-compute camera intrinsics for faster point cloud conversion"""
        hfov_rad = np.deg2rad(self.hfov_deg)
        vfov_rad = np.deg2rad(self.vfov_deg)
        
        # Calculate focal lengths
        self.fx = self.width / (2 * np.tan(hfov_rad / 2))
        self.fy = self.height / (2 * np.tan(vfov_rad / 2))
        
        # Pre-compute normalized pixel coordinates
        x = np.arange(self.width)
        y = np.arange(self.height)
        x, y = np.meshgrid(x, y)
        
        # Normalize to camera coordinates
        self.x_cam = (x - self.width / 2) / self.fx
        self.y_cam = (y - self.height / 2) / self.fy
        
    def forward(self, depth_mask_3C):
        '''
        Args:
            depth_mask_3C: torch.Tensor of shape (B, 3, H, W)
        Returns:
            point_cloud_3C: Tensor of shape [B, 3, (max_num_points+1)*max_num_persons]
        '''
        batch_size = depth_mask_3C.shape[0]
        device = depth_mask_3C.device
        dtype = depth_mask_3C.dtype
        
        # Initialize output tensor
        point_cloud_3C = torch.zeros(batch_size, 3, (self.max_num_points+1)*self.max_num_persons, 
                                   dtype=dtype, device=device)
        
        # Process all batches at once to minimize CPU-GPU transfers
        depth_maps = depth_mask_3C[:, 0]  # (B, H, W)
        indicator_maps = torch.round(depth_mask_3C[:, 1])  # (B, H, W)
        
        # Convert to numpy only once
        depth_maps_np = depth_maps.detach().cpu().numpy()
        indicator_maps_np = indicator_maps.detach().cpu().numpy()
        
        for b in range(batch_size):
            depth_map = depth_maps_np[b]
            indicator_map = indicator_maps_np[b]
            
            # Process all persons in this batch simultaneously
            self._process_batch_persons(depth_map, indicator_map, b, point_cloud_3C, device, dtype)
            
        return point_cloud_3C
    
    def _process_batch_persons(self, depth_map, indicator_map, batch_idx, point_cloud_3C, device, dtype):
        """Process all persons in a batch efficiently"""
        # Find unique person indices (excluding 0)
        indicator_map = indicator_map.astype(int)
        unique_persons = np.unique(indicator_map)
        unique_persons = unique_persons[unique_persons > 0]
        
        for person_idx in unique_persons:
            # Convert numpy scalar to Python int for tensor slicing
            person_idx = int(person_idx)
            if person_idx > self.max_num_persons:
                continue
                
            # Create person mask
            person_mask = (indicator_map == person_idx)
            
            if not np.any(person_mask):
                continue
            
            # Extract person depth with mask
            person_depth = depth_map * person_mask
            
            # Fast point cloud conversion and processing
            points = self._fast_point_cloud_processing(person_depth, person_mask)
            
            if points is not None:
                # Calculate offset and store results
                offset = (person_idx - 1) * (self.max_num_points + 1)
                point_cloud_3C[batch_idx, :, offset:offset + self.max_num_points] = torch.tensor(
                    points, dtype=dtype, device=device)
                # Add indicator point
                point_cloud_3C[batch_idx, 0, offset + self.max_num_points] = 1.0
    
    def _fast_point_cloud_processing(self, person_depth, person_mask):
        """Fast point cloud processing with vectorized operations"""
        # Apply depth threshold and mask
        valid_mask = (person_depth > 3) & person_mask
        
        if not np.any(valid_mask):
            return None
        
        # Vectorized point cloud generation
        z_cam = person_depth
        x_3d = self.x_cam * z_cam
        y_3d = self.y_cam * z_cam
        
        # Stack coordinates
        point_cloud = np.stack((x_3d, y_3d, z_cam), axis=-1)
        
        # Extract valid points using boolean indexing
        valid_points = point_cloud[valid_mask]
        
        if valid_points.shape[0] == 0:
            return None
        
        # Fast outlier removal using vectorized operations
        valid_depths = person_depth[valid_mask]
        if valid_depths.shape[0] == 0:
            return None
            
        # Calculate outlier bounds
        lower_bound, upper_bound = self._fast_outlier_removal(valid_depths)
        
        # Apply outlier filter
        outlier_mask = (valid_depths >= lower_bound) & (valid_depths <= upper_bound)
        filtered_points = valid_points[outlier_mask]
        
        if filtered_points.shape[0] == 0:
            return None
        
        # Fast sampling
        return self._fast_point_cloud_sampling(filtered_points)
    
    def _fast_outlier_removal(self, depths):
        """Fast outlier removal using numpy operations"""
        if len(depths) == 0:
            return 0, float('inf')
        
        # Use numpy's percentile for faster computation
        Q1 = np.percentile(depths, 25)
        Q3 = np.percentile(depths, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return lower_bound, upper_bound
    
    def _fast_point_cloud_sampling(self, point_cloud):
        """Fast point cloud sampling with optimized operations"""
        num_points = point_cloud.shape[0]
        
        if num_points < self.max_num_points:
            # Pad with zeros
            pad_size = self.max_num_points - num_points
            padded_points = np.zeros((self.max_num_points, 3))
            padded_points[:num_points] = point_cloud
            return padded_points.T
        else:
            # Random sampling without replacement
            indices = np.random.choice(num_points, self.max_num_points, replace=False)
            return point_cloud[indices].T


class DepthMask2PointCloudUltra(nn.Module):
    '''
    Ultra-fast version using PyTorch operations throughout to minimize CPU-GPU transfers
    '''
    def __init__(self, exp_config):
        super(DepthMask2PointCloudUltra, self).__init__()
        self.sensor_name = exp_config['sensor_name']
        self.max_num_points = exp_config['max_num_points']
        self.max_num_persons = exp_config['max_num_persons']
        
        # Pre-compute camera parameters
        if self.sensor_name == 'senxor_m08':
            self.hfov_deg = 90
            self.vfov_deg = 67
            self.height = 62
            self.width = 80
        elif self.sensor_name == 'senxor_m16':
            self.hfov_deg = 88.5
            self.vfov_deg = 58.5
            self.height = 120
            self.width = 160
        elif self.sensor_name == 'seek_thermal':
            self.hfov_deg = 81
            self.vfov_deg = 59
            self.height = 150
            self.width = 200
        else:
            raise ValueError(f"Sensor name {self.sensor_name} not supported")
        
        # Register camera intrinsics as buffers (will be moved to GPU automatically)
        self._setup_camera_intrinsics()
        
    def _setup_camera_intrinsics(self):
        """Pre-compute camera intrinsics as PyTorch tensors"""
        hfov_rad = torch.tensor(np.deg2rad(self.hfov_deg))
        vfov_rad = torch.tensor(np.deg2rad(self.vfov_deg))
        
        # Calculate focal lengths
        fx = self.width / (2 * torch.tan(hfov_rad / 2))
        fy = self.height / (2 * torch.tan(vfov_rad / 2))
        
        # Pre-compute normalized pixel coordinates
        x = torch.arange(self.width, dtype=torch.float32)
        y = torch.arange(self.height, dtype=torch.float32)
        x, y = torch.meshgrid(x, y, indexing='xy')
        
        # Normalize to camera coordinates
        self.register_buffer('x_cam', (x - self.width / 2) / fx)
        self.register_buffer('y_cam', (y - self.height / 2) / fy)
        
    def forward(self, depth_mask_3C):
        '''
        Args:
            depth_mask_3C: torch.Tensor of shape (B, 3, H, W)
        Returns:
            point_cloud_3C: Tensor of shape [B, 3, (max_num_points+1)*max_num_persons]
        '''
        batch_size = depth_mask_3C.shape[0]
        device = depth_mask_3C.device
        dtype = depth_mask_3C.dtype
        
        # Initialize output tensor
        point_cloud_3C = torch.zeros(batch_size, 3, (self.max_num_points+1)*self.max_num_persons, 
                                   dtype=dtype, device=device)
        
        # Extract depth and indicator maps
        depth_maps = depth_mask_3C[:, 0]  # (B, H, W)
        indicator_maps = torch.round(depth_mask_3C[:, 1])  # (B, H, W)
        
        # Process each batch
        for b in range(batch_size):
            depth_map = depth_maps[b]  # (H, W)
            indicator_map = indicator_maps[b]  # (H, W)
            
            # Process all persons in this batch
            self._process_batch_persons_torch(depth_map, indicator_map, b, point_cloud_3C)
            
        return point_cloud_3C
    
    def _process_batch_persons_torch(self, depth_map, indicator_map, batch_idx, point_cloud_3C):
        """Process all persons in a batch using PyTorch operations"""
        # Find unique person indices (excluding 0)
        unique_persons = torch.unique(indicator_map)
        unique_persons = unique_persons[unique_persons > 0]
        
        for person_idx in unique_persons:
            # Convert PyTorch tensor scalar to Python int for tensor slicing
            person_idx = int(person_idx.item())
            if person_idx > self.max_num_persons:
                continue
                
            # Create person mask
            person_mask = (indicator_map == person_idx)
            
            if not torch.any(person_mask):
                continue
            
            # Extract person depth with mask
            person_depth = depth_map * person_mask
            
            # Fast point cloud conversion and processing
            points = self._fast_point_cloud_processing_torch(person_depth, person_mask)
            
            if points is not None:
                # Calculate offset and store results
                offset = (person_idx - 1) * (self.max_num_points + 1)
                point_cloud_3C[batch_idx, :, offset:offset + self.max_num_points] = points
                # Add indicator point
                point_cloud_3C[batch_idx, 0, offset + self.max_num_points] = 1.0
    
    def _fast_point_cloud_processing_torch(self, person_depth, person_mask):
        """Fast point cloud processing with PyTorch operations"""
        # Apply depth threshold and mask
        valid_mask = (person_depth > 3) & person_mask
        
        if not torch.any(valid_mask):
            return None
        
        # Vectorized point cloud generation using pre-computed camera intrinsics
        z_cam = person_depth
        x_3d = self.x_cam * z_cam
        y_3d = self.y_cam * z_cam
        
        # Stack coordinates
        point_cloud = torch.stack((x_3d, y_3d, z_cam), dim=-1)  # (H, W, 3)
        
        # Extract valid points using boolean indexing
        valid_points = point_cloud[valid_mask]  # (N, 3)
        
        if valid_points.shape[0] == 0:
            return None
        
        # Fast outlier removal using PyTorch operations
        valid_depths = person_depth[valid_mask]
        if valid_depths.shape[0] == 0:
            return None
            
        # Calculate outlier bounds using PyTorch
        lower_bound, upper_bound = self._fast_outlier_removal_torch(valid_depths)
        
        # Apply outlier filter
        outlier_mask = (valid_depths >= lower_bound) & (valid_depths <= upper_bound)
        filtered_points = valid_points[outlier_mask]
        
        if filtered_points.shape[0] == 0:
            return None
        
        # Fast sampling
        return self._fast_point_cloud_sampling_torch(filtered_points)
    
    def _fast_outlier_removal_torch(self, depths):
        """Fast outlier removal using PyTorch operations"""
        if len(depths) == 0:
            return torch.tensor(0.0, device=depths.device), torch.tensor(float('inf'), device=depths.device)
        
        # Use PyTorch's quantile for faster computation
        Q1 = torch.quantile(depths, 0.25)
        Q3 = torch.quantile(depths, 0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return lower_bound, upper_bound
    
    def _fast_point_cloud_sampling_torch(self, point_cloud):
        """Fast point cloud sampling with PyTorch operations"""
        num_points = point_cloud.shape[0]
        
        if num_points < self.max_num_points:
            # Pad with zeros
            padded_points = torch.zeros(self.max_num_points, 3, device=point_cloud.device, dtype=point_cloud.dtype)
            padded_points[:num_points] = point_cloud
            return padded_points.T
        else:
            # Random sampling without replacement
            indices = torch.randperm(num_points, device=point_cloud.device)[:self.max_num_points]
            return point_cloud[indices].T


class DepthMaskMetrics(nn.Module):
    def __init__(self, exp_config):
        """
        Args:
            binary_threshold (float): Threshold for converting the predicted binary mask 
                                      to a binary decision (default 0.5).
        """
        super(DepthMaskMetrics, self).__init__()
        self.binary_threshold = exp_config['detection_threshold']

    def forward(self, pred, target):
        """
        Computes the metrics for depth, indicator, and binary segmentation.
        
        Args:
            pred (torch.Tensor): Predicted 3-channel output of shape (B, 3, H, W).
            target (torch.Tensor): Ground truth 3-channel output of shape (B, 3, H, W).
        
        Returns:
            dict: Dictionary containing various evaluation metrics.
        """
        # Split channels (each of shape (B, H, W))
        pred_depth = pred[:, 0, :, :]
        pred_indicator = pred[:, 1, :, :]
        pred_binary = pred[:, 2, :, :]

        target_depth = target[:, 0, :, :]
        target_indicator = target[:, 1, :, :]
        target_binary = target[:, 2, :, :]
        
        epsilon = 1e-8

        # 1. Depth Metrics
        depth_mae = torch.mean(torch.abs(pred_depth - target_depth))

        # 2. Indicator Map Metrics (Detection)
        # Round predicted indicator values to the nearest integer
        pred_indicator_class = torch.round(pred_indicator)
        
        # indicator_correct = (pred_indicator_class == target_indicator).float()
        # indicator_accuracy = indicator_correct.mean()
        # Calculate precision, recall, F1 (treating indicator as detection)
        classes = torch.unique(target_indicator)  # classes are actually the user_ids in the map
        # leave the 0 out, which is the backgorund
        classes = classes[classes != 0]
        
        class_precision_list = []
        class_recall_list = []
        class_f1_list = []
        for class_idx in classes:
            class_mask = (target_indicator == class_idx)
            class_pred = (pred_indicator_class == class_idx)
            # Calculate true positives (both class_mask and class_pred are 1)
            true_positives = (class_mask & class_pred).float().sum()            
            # Calculate false positives (class_pred is 1 but class_mask is 0)
            false_positives = (class_pred & ~class_mask).float().sum()
            # Calculate false negatives (class_mask is 1 but class_pred is 0)
            false_negatives = (~class_pred & class_mask).float().sum()
            
            # Calculate class_correct (where both class_mask and class_pred are 1)
            class_precision = true_positives / (true_positives + false_positives + epsilon)
            class_recall = true_positives / (true_positives + false_negatives + epsilon)
            class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall + epsilon)
            class_precision_list.append(class_precision)
            class_recall_list.append(class_recall)
            class_f1_list.append(class_f1)
        detection_precision = torch.tensor(class_precision_list).mean()
        detection_recall = torch.tensor(class_recall_list).mean()
        detection_f1 = torch.tensor(class_f1_list).mean()
            
        # 3. Binary (forground and background) Segmentation Metrics
        pred_binary_thresh = (pred_binary > self.binary_threshold).float()
        intersection = (pred_binary_thresh * target_binary).sum(dim=(1, 2))
        union = ((pred_binary_thresh + target_binary) > 0).float().sum(dim=(1, 2))
        iou = (intersection + epsilon) / (union + epsilon)
        binary_iou = iou.mean()

        return {
            'depth_mae': depth_mae.item(),
            # 'detection_accuracy': indicator_accuracy.item(),
            'detection_precision': detection_precision.item(),
            'detection_recall': detection_recall.item(),
            'detection_f1': detection_f1.item(),
            'binary_iou': binary_iou.item(),
        }


class PointCloudMetrics(nn.Module):
    def __init__(self, exp_config):
        """
        Args:
            exp_config (dict): Experiment configuration
        """
        super(PointCloudMetrics, self).__init__()
        self.max_num_points = exp_config['max_num_points']  
        self.max_num_persons = exp_config['max_num_persons'] 
        self.detection_threshold = exp_config['detection_threshold'] # Threshold for person detection confidence
        self.point_threshold = exp_config['point_threshold']  # Distance threshold for point accuracy calculation
        self.points_per_person = self.max_num_points + 1  # +1 for indicator point
    
    def forward(self, predictions, targets):
        """
        Compute comprehensive metrics for 3D point cloud predictions.
        
        Args:
            predictions: Tensor of shape [B, 3, (max_num_points+1)*max_num_persons]
            targets: Tensor of shape [B, 3, (max_num_points+1)*max_num_persons]
            
        Returns:
            dict: Dictionary containing various evaluation metrics
        """
        # Extract indicators and extract metrics
        target_indicator, target_cloud_point, pred_indicator, pred_cloud_point, present_persons = self._data_parsing(predictions, targets)
        '''
        target_indicator: shape: [B, max_num_persons]
        target_cloud_point: shape: [B, 3, max_num_points, max_num_persons]
        pred_indicator: shape: [B, max_num_persons]
        pred_cloud_point: shape: [B, 3, max_num_points, max_num_persons]
        present_persons: list of indices of the present persons, when batch_size == 1
        '''
        chamfer_dist = self._chamfer_distance2(pred_cloud_point, target_cloud_point, target_indicator)
            
        detection_metrics = self._calculate_detection_metrics(pred_indicator, target_indicator)
        # nearest_neighbor_dist = self._nearest_neighbor_distance(pred_cloud_point, target_cloud_point)
        
        # Combine all metrics into a single dictionary
        metrics = {
            'detection_precision': detection_metrics['precision'].item(),
            'detection_recall': detection_metrics['recall'].item(),
            'detection_f1': detection_metrics['f1'].item(),
            'chamfer_distance': chamfer_dist.item(),
            # 'nearest_neighbor_distance': nearest_neighbor_dist.item(),
        }
        return metrics

    
    def _data_parsing(self, predictions, targets):
        """
        Parse the predictions and targets into point clouds and indicators
        """
        batch_size = predictions.shape[0]
        points_per_person = self.max_num_points + 1
        
        target_indicator = torch.zeros(batch_size, self.max_num_persons, device=targets.device)        
        target_cloud_point = torch.zeros(batch_size, 3, self.max_num_points, self.max_num_persons, 
                                         device=targets.device)
        pred_indicator = torch.zeros(batch_size, self.max_num_persons, device=predictions.device)
        pred_cloud_point = torch.zeros(batch_size, 3, self.max_num_points, self.max_num_persons, 
                                         device=targets.device)
        for p in range(self.max_num_persons):
            # Get the index of the indicator point for this person
            indicator_idx = (p + 1) * points_per_person - 1
            # Check if this person is present (indicator value > 0.5)
            target_indicator[:, p] = (targets[:, 0, indicator_idx] > 0.5).float()
            pred_indicator[:, p] = predictions[:, 0, indicator_idx]
            # Extract the point cloud for this person (excluding the indicator point)
            start_idx = p * points_per_person
            end_idx = start_idx + self.max_num_points
            target_cloud_point[:, :, :, p] = targets[:, :, start_idx:end_idx]
            pred_cloud_point[:, :, :, p] = predictions[:, :, start_idx:end_idx]
        
        present_persons = []
        if batch_size == 1: # for single frame, get the index of person that exist
            for p in range(self.max_num_persons):
                if target_indicator[0, p] > 0:
                    present_persons.append(p)
        return target_indicator, target_cloud_point, pred_indicator, pred_cloud_point, present_persons
    
    
    def _calculate_detection_metrics(self, pred_indicator, target_indicator):
        """Calculate precision, recall, and F1 for person detection"""
        # Binarize predictions using threshold
        pred_detected = (pred_indicator > self.detection_threshold).float()
        gt_detected = (target_indicator > self.detection_threshold).float()
        
        # Calculate true positives, false positives, false negatives
        true_positives = (pred_detected * gt_detected).sum()
        false_positives = (pred_detected * (1 - gt_detected)).sum()
        false_negatives = ((1 - pred_detected) * gt_detected).sum()
        
        # Calculate precision, recall, F1
        epsilon = 1e-8  # Avoid division by zero
        precision = true_positives / (true_positives + false_positives + epsilon)
        recall = true_positives / (true_positives + false_negatives + epsilon)
        f1 = 2 * (precision * recall) / (precision + recall + epsilon)
        
        return {'precision': precision, 'recall': recall, 'f1': f1}
    
    
    def _chamfer_distance2(self, predicted_points, target_points, target_indicator):
        """
        Calculate the Chamfer distance between two point clouds using pytorch3d with proper handling of padding.
        Uses x_lengths and y_lengths to handle both 0 padding and present persons only.
        
        Args:
            predicted_points: Tensor of shape [B, 3, N, max_num_persons]
            target_points: Tensor of shape [B, 3, N, max_num_persons]
            target_indicator: Tensor of shape [B, max_num_persons] indicating which persons are present
        Returns:
            The mean Chamfer distance
        """
        batch_size = predicted_points.shape[0]
        
        # Reshape to [B*max_num_persons, N, 3] for easier processing
        pred_reshaped = predicted_points.view(batch_size * self.max_num_persons, self.max_num_points, 3)
        target_reshaped = target_points.view(batch_size * self.max_num_persons, self.max_num_points, 3)
        
        # Create mask for valid points (non-zero coordinates)
        pred_valid_mask = torch.any(pred_reshaped != 0, dim=2)  # [B*max_num_persons, N]
        target_valid_mask = torch.any(target_reshaped != 0, dim=2)  # [B*max_num_persons, N]
        
        # Get lengths for each point cloud
        pred_lengths = pred_valid_mask.sum(dim=1)  # [B*max_num_persons]
        target_lengths = target_valid_mask.sum(dim=1)  # [B*max_num_persons]
        
        # Create mask for present persons
        present_mask = target_indicator.view(-1) > 0.5  # [B*max_num_persons]
        
        # Filter to only present persons with valid points
        valid_mask = present_mask & (pred_lengths > 0) & (target_lengths > 0)
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=predicted_points.device, requires_grad=True)
        
        # Get valid point clouds and their lengths
        valid_pred = pred_reshaped[valid_mask]  # [num_valid, N, 3]
        valid_target = target_reshaped[valid_mask]  # [num_valid, N, 3]
        valid_pred_lengths = pred_lengths[valid_mask]  # [num_valid]
        valid_target_lengths = target_lengths[valid_mask]  # [num_valid]
        
        # Calculate chamfer distance with lengths
        chamfer_dist, _ = chamfer_distance(
            valid_pred, 
            valid_target,
            x_lengths=valid_pred_lengths,
            y_lengths=valid_target_lengths,
            batch_reduction="mean",
            point_reduction="mean"
        )
        return chamfer_dist
    
    
    def _nearest_neighbor_distance(self, predicted_points, target_points, k=3):
        """
        Calculate average distance to k nearest neighbors
        """
        batch_size = predicted_points.shape[0]
        pred_p = predicted_points.view(batch_size, 3, -1).permute(0, 2, 1)
        target_p = target_points.view(batch_size, 3, -1).permute(0, 2, 1)
        
        # Calculate distances to k nearest neighbors
        distances = []
        for i in range(batch_size):
            # Use KNN instead of all-pairs distance
            dist, _ = torch.cdist(pred_p[i:i+1], target_p[i:i+1], p=2).topk(k, largest=False, dim=2)
            distances.append(dist.mean())
        
        return torch.stack(distances).mean()
    
    def calculate_hausdorff_distance(self, predicted_points, target_points, target_indicator):
        """
        Calculate the Hausdorff distance between two point clouds using pytorch3d with proper handling of padding.
        Uses x_lengths and y_lengths to handle both 0 padding and present persons only.
        
        Args:
            predicted_points: Tensor of shape [B, 3, N, max_num_persons]
            target_points: Tensor of shape [B, 3, N, max_num_persons]
            target_indicator: Tensor of shape [B, max_num_persons] indicating which persons are present
        Returns:
            The mean Chamfer distance
        """
        batch_size = predicted_points.shape[0]
        
        # Reshape to [B*max_num_persons, N, 3] for easier processing
        pred_reshaped = predicted_points.view(batch_size * self.max_num_persons, self.max_num_points, 3)
        target_reshaped = target_points.view(batch_size * self.max_num_persons, self.max_num_points, 3)
        
        # Create mask for valid points (non-zero coordinates)
        pred_valid_mask = torch.any(pred_reshaped != 0, dim=2)  # [B*max_num_persons, N]
        target_valid_mask = torch.any(target_reshaped != 0, dim=2)  # [B*max_num_persons, N]
        
        # Get lengths for each point cloud
        pred_lengths = pred_valid_mask.sum(dim=1)  # [B*max_num_persons]
        target_lengths = target_valid_mask.sum(dim=1)  # [B*max_num_persons]
        
        # Create mask for present persons
        present_mask = target_indicator.view(-1) > 0.5  # [B*max_num_persons]
        
        # Filter to only present persons with valid points
        valid_mask = present_mask & (pred_lengths > 0) & (target_lengths > 0)
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=predicted_points.device, requires_grad=True)
        
        # Get valid point clouds and their lengths
        valid_pred = pred_reshaped[valid_mask]  # [num_valid, N, 3]
        valid_target = target_reshaped[valid_mask]  # [num_valid, N, 3]
        valid_pred_lengths = pred_lengths[valid_mask]  # [num_valid]
        valid_target_lengths = target_lengths[valid_mask]  # [num_valid]
        
        # Calculate chamfer distance with lengths
        hausdorff_dist, _ = chamfer_distance(
            valid_pred, 
            valid_target,
            x_lengths=valid_pred_lengths,
            y_lengths=valid_target_lengths,
            batch_reduction="mean",
            point_reduction="max", # this will lead to the Hausdorff distance
        )
        return hausdorff_dist
    
    def calculate_point2point_distance(self, predicted_points, target_points, target_indicator):
        """
        Calculate the point-to-point distance between two point clouds
        Args:
            predicted_points: Tensor of shape [B, 3, N, max_num_persons]
            target_points: Tensor of shape [B, 3, N, max_num_persons]
            target_indicator: Tensor of shape [B, max_num_persons] indicating which persons are present
        Returns:
            The mean Chamfer distance
        """
        batch_size = predicted_points.shape[0]
        
        # Reshape to [B*max_num_persons, N, 3] for easier processing
        pred_reshaped = predicted_points.view(batch_size * self.max_num_persons, self.max_num_points, 3)
        target_reshaped = target_points.view(batch_size * self.max_num_persons, self.max_num_points, 3)
        
        # Create mask for valid points (non-zero coordinates)
        pred_valid_mask = torch.any(pred_reshaped != 0, dim=2)  # [B*max_num_persons, N]
        target_valid_mask = torch.any(target_reshaped != 0, dim=2)  # [B*max_num_persons, N]
        # the common valid mask
        valid_mask = pred_valid_mask & target_valid_mask
        
        # point to point distance
        point_to_point_dist = torch.linalg.norm(pred_reshaped - target_reshaped, dim=2, ord=2) # [B*max_num_persons, N]
        
        # Create mask for present persons
        present_mask = target_indicator.view(-1) > 0.5  # [B*max_num_persons]
        # expand to [B*max_num_persons, N]
        present_mask = present_mask.unsqueeze(1).expand(-1, self.max_num_points) # [B*max_num_persons, N]
        
        # overall valid mask
        valid_mask = valid_mask & present_mask  # [B*max_num_persons, N]
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=predicted_points.device, requires_grad=True)
        
        # calculate the mean point to point distance
        point_to_point_dist = point_to_point_dist[valid_mask] # [num_valid]
        point_to_point_dist = point_to_point_dist.mean() # [1]
        return point_to_point_dist
        

