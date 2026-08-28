import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
# from monai.losses import DiceLoss
from pytorch3d.loss import chamfer_distance
import math

class CombinedLoss(nn.Module):
    def __init__(self, exp_config, debug=False):
        super(CombinedLoss, self).__init__()
        self.depth_loss_weight = exp_config.get('depth_loss_weight', 1.0)       # for the multi-primitive depth output
        self.indicator_loss_weight = exp_config.get('indicator_loss_weight', 1.0)    # for the user index map
        self.OAV_binary_loss_weight = exp_config.get('OAV_binary_loss_weight', 1.0)    # for OAV refinement module
        self.BEV_binary_loss_weight = exp_config.get('BEV_binary_loss_weight', 1.0)    # for BEV refinement module
        self.reconstruction_loss_weight = exp_config.get('reconstruction_loss_weight', 1.0)    # for reconstruction module
        self.debug = debug
        
        enable_aov_refine = exp_config.get('enable_aov_refine', True)
        enable_bev_refine = exp_config.get('enable_bev_refine', True)
        enable_reconstruction = exp_config.get('enable_reconstruction', True)
        reconstruction_loss_target_only = exp_config.get('reconstruction_loss_target_only', True)  # the reconstruction loss is only calculated on the target pixels
        self.enable_aov_refine = enable_aov_refine
        self.enable_bev_refine = enable_bev_refine
        # make sure the following param. for the DIM is identical to those in the model setting
        self.depth_max = 8000.
        self.bin_size = 20.
        self.sigma = .5
        self.enable_reconstruction = enable_reconstruction
        self.reconstruction_loss_target_only = reconstruction_loss_target_only
        self.depth_loss_fn = nn.HuberLoss()  # Huber Loss for depth map
        self.indicator_loss_fn = nn.HuberLoss()  # Huber Loss for single-channel person ID prediction (better than MSE for discrete values)
        self.OAV_binary_loss_fn = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss (since we apply sigmoid manually)
        self.BEV_binary_loss_fn = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss (since we apply sigmoid manually)
        self.reconstruction_loss_fn = nn.HuberLoss()  # Huber Loss for reconstruction module
    
    def forward(self, pred, target, input_tensor):
        """
            target (torch.Tensor): Ground truth 3-channel depth mask of shape (B, 3, H, W).
            input (torch.Tensor): Input tensor of shape (B, 1, H, W).
        """
        if target.shape[1] != 3:
            raise ValueError(f"Target depth mask must have 3 channels, but got {target.shape[1]} channels")
        # Split ground truth target into individual channels
        target_depth = target[:, 0, :, :]  # Depth map (B, H, W)
        target_indicator = target[:, 1, :, :]  # Indicator map (B, H, W), where 0 indicates no object and 1 indicates first person, 2 indicates second person, ...
        target_binary = target[:, 2, :, :]  # Binary mask (B, H, W)
        # Clamp target_binary to [0,1] to ensure it's valid for BCEWithLogitsLoss
        target_binary = torch.clamp(target_binary, 0.0, 1.0)
        input_tensor = input_tensor.detach().squeeze(1)
        
        if not self.enable_reconstruction:
            pred_depth = pred['depth'].squeeze(1)
            pred_indicator = pred['user_index'].squeeze(1)
            # only calculate the depth loss, indicator loss
            depth_loss = self.depth_loss_fn(pred_depth, target_depth)
            depth_loss_rescale = depth_loss / 100.0  # Scale down depth loss
            indicator_loss = self.indicator_loss_fn(pred_indicator, target_indicator)
            total_loss = (
                self.depth_loss_weight * depth_loss_rescale
                + self.indicator_loss_weight * indicator_loss
            )
            
            if self.debug:
                print(f"Depth Loss: {depth_loss.item():.6f}, Rescaled Depth Loss: {depth_loss_rescale.item():.6f}, Indicator Loss: {indicator_loss.item():.6f}")
            
            return total_loss
        else:
            # calculate the depth loss, indicator loss, binary loss, OAV binary loss, BEV binary loss, and reconstruction loss
            pred_depth = pred['depth'].squeeze(1)
            depth_loss = self.depth_loss_fn(pred_depth, target_depth)
            depth_loss_rescale = depth_loss / 100.0  # Scale down depth loss
            pred_indicator = pred['user_index'].squeeze(1)
            indicator_loss = self.indicator_loss_fn(pred_indicator, target_indicator)
            pred_reconstruction = pred['Tp_recon'].squeeze(1)
            # Only calculate reconstruction loss where target_binary == 1
            if self.reconstruction_loss_target_only:
                mask = (target_binary > 0.8).float()
                mask = mask.bool()
            else:
                mask = torch.ones_like(target_binary).bool()
            
            if mask.sum() > 0:
                # Normal reconstruction loss when we have valid pixels
                if self.debug:
                    print(f"Reconstruction loss: {pred_reconstruction[mask].shape}, {input_tensor[mask].shape}")
                reconstruction_loss = self.reconstruction_loss_fn(
                    pred_reconstruction[mask], 
                    input_tensor[mask]
                )
                rescaled_reconstruction_loss = reconstruction_loss / 100.0
            else:
                if self.debug:
                    print("No valid pixels for reconstruction loss")
                # Fallback: use a small regularization loss on the constants to ensure gradient flow
                # This prevents the map tensors from having zero gradients
                const_regularization = 1e-6 * (
                    pred['depth'].mean() + 
                    pred['Emissivity'].mean() + 
                    pred['Reflectance'].mean() + 
                    pred['Temperature'].mean()
                )
                rescaled_reconstruction_loss = const_regularization
            
            if self.enable_aov_refine:
                pred_aov_binary = pred['aov_output'].squeeze(1)
                aov_binary_loss = self.OAV_binary_loss_fn(pred_aov_binary, target_binary)
            if self.enable_bev_refine:
                pred_bev_binary = pred['bev_output'].squeeze(1)
                target_bev = DIM_backprop(target_depth, depth_max=self.depth_max, bin_size=self.bin_size, sigma=self.sigma, normalize=True, chunk_h=None)
                bev_binary_loss = self.BEV_binary_loss_fn(pred_bev_binary, target_bev)
            
            if not self.enable_aov_refine:
                aov_binary_loss = torch.tensor(0.0, device=depth_loss_rescale.device, requires_grad=True)
            if not self.enable_bev_refine:
                bev_binary_loss = torch.tensor(0.0, device=depth_loss_rescale.device, requires_grad=True)
            
            total_loss = (
                self.depth_loss_weight * depth_loss_rescale
                + self.indicator_loss_weight * indicator_loss
                + self.OAV_binary_loss_weight * aov_binary_loss
                + self.BEV_binary_loss_weight * bev_binary_loss
                + self.reconstruction_loss_weight * rescaled_reconstruction_loss
            )
            if self.debug:
                print(f"Depth Loss: {depth_loss.item():.6f}, Rescaled Depth Loss: {depth_loss_rescale.item():.6f}, Indicator Loss: {indicator_loss.item():.6f}, AOV Binary Loss: {aov_binary_loss.item():.6f}, BEV Binary Loss: {bev_binary_loss.item():.6f}, Reconstruction Loss: {reconstruction_loss.item():.6f}, Rescaled Reconstruction Loss: {rescaled_reconstruction_loss.item():.6f}")
            return total_loss
                

# copy from the model implementation file
def DIM_backprop(
    depth: torch.Tensor,           # (B, H, W), same unit as depth_max/sigma (e.g., mm)
    depth_max: float = 4000.0,
    bin_size: float = 1.0,
    sigma: float = 5.0,            # can also be a Tensor broadcastable to (B,H,W,1)
    normalize: bool = True,
    chunk_h: int = None,
    treat_zero_invalid: bool = True,   # zero (or near-zero) depth contributes nothing
    zero_eps: float = 1e-6,            # threshold for “zero”
    debug: bool = False,
    ) -> torch.Tensor:
    """
    Differentiable Index Mapping (OAV Depth -> BEV soft histogram).

    Returns:
        out: (B, W, D) soft histogram. Fully differentiable w.r.t. `depth` and `sigma`.
    Notes:
        - Do NOT hard-threshold `out` inside this function if you want gradients.
          For visualization, do e.g. `out_vis = (out > 0.5).float().detach()`.
        - Use `chunk_h` to control memory: larger chunks use more memory but run faster.
    """
    assert depth.ndim == 3, "depth must be (B,H,W)"
    assert depth_max > 0 and bin_size > 0 and sigma is not None
    B, H, W = depth.shape
    D = int(depth_max / bin_size) + 1

    device, dtype = depth.device, depth.dtype
    eps = 1e-8
    SQRT_2PI = math.sqrt(2.0 * math.pi)

    bins = torch.arange(D, device=device, dtype=dtype) * float(bin_size)  # (D,)

    if chunk_h is None:
        chunk_h = H

    # allow scalar or tensor sigma
    if torch.is_tensor(sigma):
        # reshape to broadcast with d: (B,h,W,1) later
        sigma_is_tensor = True
    else:
        sigma_is_tensor = False
        sigma = torch.tensor(float(sigma), device=device, dtype=dtype)

    out = None  # accumulate functionally to keep autograd graph

    for y0 in range(0, H, chunk_h):
        y1 = min(H, y0 + chunk_h)
        d = depth[:, y0:y1, :, None]  # (B, h, W, 1)

        # validity mask
        valid = torch.isfinite(d) & (d <= depth_max)
        if treat_zero_invalid:
            valid &= (d > zero_eps)
        else:
            valid &= (d >= 0)

        # broadcast sigma
        if sigma_is_tensor:
            # expect sigma broadcastable to (B, h, W, 1)
            sig = sigma[..., None] if sigma.ndim == 3 else sigma
            sig = sig.to(device=device, dtype=dtype)
            sig = torch.broadcast_to(sig, d.shape)
        else:
            sig = sigma  # scalar tensor

        # Gaussian kernel
        # shape math: bins[None,None,None,:] - d => (B,h,W,D)
        z = (bins.view(1, 1, 1, D) - d) / (sig + eps)
        g = torch.exp(-0.5 * z * z)
        if normalize:
            g = g / ((sig + eps) * SQRT_2PI)  # proper 1D Gaussian normalization

        # zero-out invalids (keeps gradients where valid)
        g = g * valid  # broadcast bool -> float

        if debug:
            with torch.no_grad():
                g_min = float(g.min().item()) if g.numel() else float("nan")
                g_max = float(g.max().item()) if g.numel() else float("nan")
                valid_frac = float(valid.float().mean().item())
                print(f"[DIM_v2_backprop] rows {y0}:{y1} | g_min={g_min:.6g} "
                      f"g_max={g_max:.6g} | valid_frac={valid_frac:.3f}")

        term = g.sum(dim=1)  # (B, W, D) aggregate along height
        out = term if out is None else (out + term)  # functional accumulation

    return out  # (B, W, D)
        

class DepthMaskLoss(nn.Module):
    def __init__(self,exp_config, debug=False):
        super(DepthMaskLoss, self).__init__()
        self.depth_loss_weight = exp_config.get('depth_loss_weight', 1.0)
        self.indicator_loss_weight = exp_config.get('indicator_loss_weight', 1.0)
        self.binary_loss_weight = exp_config.get('binary_loss_weight', 1.0)
        self.debug = debug

        # Loss functions
        # - L1Loss for depth: Robust to outliers, good for continuous depth regression
        # - L1Loss for indicator: Better for discrete person ID prediction (less sensitive to outliers than MSE)
        # - DiceLoss for binary mask: Handles class imbalance in segmentation tasks
        # Note: Depth loss is normalized by dividing by 100 to balance with other losses
        
        # self.depth_loss_fn = nn.L1Loss()  # L1 Loss for depth map
        # self.indicator_loss_fn = nn.L1Loss()  # L1 Loss for single-channel person ID prediction (better than MSE for discrete values)
        # use huber losses for the above two losses
        self.depth_loss_fn = nn.HuberLoss()  # Huber Loss for depth map
        self.indicator_loss_fn = nn.HuberLoss()  # Huber Loss for single-channel person ID prediction (better than MSE for discrete values)
        # self.binary_loss_fn = DiceLoss()  # Dice Loss for binary mask
        self.binary_loss_fn = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss (since we apply sigmoid manually)

    def forward(self, pred, target):
        """
        Compute the loss for the 3-channel depth mask.
        Args:
            pred (torch.Tensor): Predicted 3-channel depth mask of shape (B, 3, H, W).
            target (torch.Tensor): Ground truth 3-channel depth mask of shape (B, 3, H, W).
        Returns:
            loss (torch.Tensor): Combined loss value.
        """
        if pred.shape[1] != 3:
            raise ValueError(f"Predicted depth mask must have 3 channels, but got {pred.shape[1]} channels")
        if target.shape[1] != 3:
            raise ValueError(f"Target depth mask must have 3 channels, but got {target.shape[1]} channels")
        
        if pred.shape != target.shape:
            raise ValueError(f"Predicted and target depth mask must have the same shape, but got {pred.shape} and {target.shape}")
        
        # Split the prediction and target into individual channels
        pred_depth = pred[:, 0, :, :]  # Depth map (B, H, W), 0 means no object, otherwise depth value of the persons
        pred_indicator = pred[:, 1, :, :]  # Indicator map (B, H, W), where 0 indicates no object and 1 indicates first person, 2 indicates second person, ...
        pred_binary = pred[:, 2, :, :]  # Binary mask (B, H, W), 0 means background, 1 means person

        target_depth = target[:, 0, :, :]  # Depth map (B, H, W)
        target_indicator = target[:, 1, :, :]  # Indicator map (B, H, W), where 0 indicates no object and 1 indicates first person, 2 indicates second person, ...
        target_binary = target[:, 2, :, :]  # Binary mask (B, H, W)

        # Clamp target_binary to [0,1] to ensure it's valid for BCEWithLogitsLoss
        target_binary = torch.clamp(target_binary, 0.0, 1.0)

        # Compute depth loss (L1 Loss)
        depth_loss = self.depth_loss_fn(pred_depth, target_depth)

        # Compute indicator loss (L1 Loss for single-channel person ID prediction)
        # Both pred_indicator and target_indicator have shape (B, H, W)
        # pred_indicator: continuous values that should approximate person IDs (0, 1, 2, ...)
        # target_indicator: discrete person IDs (0, 1, 2, ...)
        indicator_loss = self.indicator_loss_fn(pred_indicator, target_indicator)

        # Compute binary mask loss (BCE Loss)
        # Apply sigmoid to predictions since model outputs logits but targets are probabilities
        binary_loss = self.binary_loss_fn(pred_binary, target_binary)
        
        
        # Normalize losses to similar scales for better training stability
        # Depth loss is typically much larger, so we scale it down
        depth_loss_normalized = depth_loss / 100.0  # Scale down depth loss
        
        # Monitor individual losses for debugging training instability
        # print(f"Depth Loss: {depth_loss.item():.6f}, Normalized Depth Loss: {depth_loss_normalized.item():.6f}, Indicator Loss: {indicator_loss.item():.6f}, Binary Loss: {binary_loss.item():.6f}")
        if self.debug:
            print(f"Depth Loss: {depth_loss.item():.6f}, Normalized Depth Loss: {depth_loss_normalized.item():.6f}, Indicator Loss: {indicator_loss.item():.6f}, Binary Loss: {binary_loss.item():.6f}")
        # Combine losses with weights
        total_loss = (
            self.depth_loss_weight * depth_loss_normalized
            + self.indicator_loss_weight * indicator_loss
            + self.binary_loss_weight * binary_loss
        )

        return total_loss



# Add the following class for point cloud loss calculation
class PointCloudLoss(nn.Module):
    def __init__(self, exp_config, loss_type='chamfer_pytorch3d'):
        """
        Loss function for point cloud data generated with 'point_cloud_3C' label type.
        
        Args:
            max_num_points (int): Maximum number of points per person
            max_num_persons (int): Maximum number of persons in the environment
            loss_type (str): Type of loss to use: 'chamfer', 'mse', or 'combined'
        """
        super(PointCloudLoss, self).__init__()
        self.max_num_points = exp_config['max_num_points']
        self.max_num_persons = exp_config['max_num_persons']
        self.loss_type = loss_type
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        # self.binary_loss_fn = DiceLoss()
        self.binary_loss_fn = nn.BCEWithLogitsLoss()
    
    
    def _chamfer_distance(self, predicted_points, target_points):
        """
        Calculate the Chamfer distance between two point clouds with reduced memory usage.
        
        Args:
            predicted_points: Tensor of shape [B, 3, N, max_num_persons]
            target_points: Tensor of shape [B, 3, N, max_num_persons]            
        Returns:
            The mean Chamfer distance
        """
        batch_size = predicted_points.shape[0]
        
        # Reshape and permute to [B, N_total, 3]
        pred_p = predicted_points.view(batch_size, 3, -1).permute(0, 2, 1)  # [B, N_total, 3]
        target_p = target_points.view(batch_size, 3, -1).permute(0, 2, 1)   # [B, N_total, 3]
        
        # Initialize variables to accumulate distances
        total_min_pred_to_target = 0
        total_min_target_to_pred = 0
        
        # Process in chunks to reduce memory usage
        chunk_size = 1024  # Adjust this based on your GPU memory
        
        # Process pred_p to target_p distances in chunks
        for i in range(0, pred_p.shape[1], chunk_size):
            pred_chunk = pred_p[:, i:i+chunk_size, :]
            dist_matrix = torch.cdist(pred_chunk, target_p, p=2)  # [B, chunk_size, N_target]
            min_distances, _ = torch.min(dist_matrix, dim=2)  # [B, chunk_size]
            total_min_pred_to_target += min_distances.sum(dim=1)  # [B]
            
            # Free memory
            del dist_matrix, min_distances
            torch.cuda.empty_cache()
        
        # Process target_p to pred_p distances in chunks
        for i in range(0, target_p.shape[1], chunk_size):
            target_chunk = target_p[:, i:i+chunk_size, :]
            dist_matrix = torch.cdist(target_chunk, pred_p, p=2)  # [B, chunk_size, N_pred]
            min_distances, _ = torch.min(dist_matrix, dim=2)  # [B, chunk_size]
            total_min_target_to_pred += min_distances.sum(dim=1)  # [B]
            
            # Free memory
            del dist_matrix, min_distances
            torch.cuda.empty_cache()
        
        # Calculate final averages
        avg_pred_to_target = total_min_pred_to_target / pred_p.shape[1]
        avg_target_to_pred = total_min_target_to_pred / target_p.shape[1]
        
        # Return mean chamfer distance
        chamfer_dist = (avg_pred_to_target + avg_target_to_pred) / 2
        
        return chamfer_dist.mean()
    
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
        pred_valid_mask = torch.any((pred_reshaped > -1e-6) & (pred_reshaped < 1e6), dim=2)  # [B*max_num_persons, N]
        target_valid_mask = torch.any((target_reshaped > -1e-6) & (target_reshaped < 1e6), dim=2)  # [B*max_num_persons, N]
        
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
        

    def forward(self, predictions, targets):
        """
        Calculate the loss between predicted and target point clouds.
        
        Args:
            predictions: Tensor of shape [B, 3, (max_num_points+1)*max_num_persons]
            targets: Tensor of shape [B, 3, (max_num_points+1)*max_num_persons]
            
        Returns:
            The loss value
        """
        batch_size = predictions.shape[0]
        points_per_person = self.max_num_points + 1
        
        # Extract indicator points to determine which persons are present in the target
        target_indicator = torch.zeros(batch_size, self.max_num_persons, device=targets.device)        
        # Extract point clouds for each person
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
        
        # calculate the detection loss
        detection_loss = self.binary_loss_fn(pred_indicator, target_indicator)
        # calculate the point cloud position loss
        if self.loss_type == 'mse':
            point_cloud_loss = self.mse_loss(pred_cloud_point, target_cloud_point)
        elif self.loss_type == 'chamfer':
            point_cloud_loss = self._chamfer_distance(pred_cloud_point, target_cloud_point)
        elif self.loss_type == 'chamfer_pytorch3d':
            point_cloud_loss = self._chamfer_distance2(pred_cloud_point, target_cloud_point, target_indicator)
        elif self.loss_type == 'combined':
            point_cloud_loss = self.mse_loss(pred_cloud_point, target_cloud_point) + self._chamfer_distance(pred_cloud_point, target_cloud_point)
        elif self.loss_type == 'combined_pytorch3d':
            point_cloud_loss = self.mse_loss(pred_cloud_point, target_cloud_point) + self._chamfer_distance2(pred_cloud_point, target_cloud_point, target_indicator)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}") 
        # print(f"Detection Loss: {detection_loss.item()}, Point Cloud Loss: {point_cloud_loss.item()}")
        return detection_loss + point_cloud_loss
          
    
    
if __name__ == "__main__":
    import time
    
    # Test configuration
    exp_config = {
        'depth_loss_weight': 1.0,
        'indicator_loss_weight': 1.0,
        'binary_loss_weight': 1.0,
        'max_num_points': 3000,
        'max_num_persons': 5
    }
    
    # Test DepthMaskLoss
    print("\n=== Testing DepthMaskLoss ===")
    # Create dummy prediction and target for depth mask (3 channels: depth, indicator, binary)
    pred_depth = torch.randn(2, 3, 256, 256)  # batch_size=2, 3 channels, 256x256
    # Make target more realistic:
    # - Depth channel: positive values
    # - Indicator channel: values between 0 and max_num_persons
    # - Binary channel: values between 0 and 1
    target_depth = torch.abs(torch.randn(2, 3, 256, 256))
    target_depth[:, 1, :, :] = target_depth[:, 1, :, :] * exp_config['max_num_persons']  # Scale indicator channel
    target_depth[:, 2, :, :] = torch.sigmoid(target_depth[:, 2, :, :])  # Binary channel between 0 and 1
    
    # Initialize and test DepthMaskLoss
    depth_loss_fn = DepthMaskLoss(exp_config)
    start_time = time.time()
    depth_loss = depth_loss_fn(pred_depth, target_depth)
    depth_time = time.time() - start_time
    print(f"DepthMaskLoss: {depth_loss.item():.4f}")
    print(f"Calculation time: {depth_time:.4f} seconds")
    
    # Test PointCloudLoss
    print("\n=== Testing PointCloudLoss ===")
    # Create dummy prediction and target for point cloud
    total_points = (exp_config['max_num_points'] + 1) * exp_config['max_num_persons']
    pred_points = torch.randn(32, 3, total_points, requires_grad=True).cuda()  # batch_size=2, 3 coordinates, total_points
    # target_points = torch.abs(torch.randn(2, 3, total_points)).cuda()  # Make target points positive
    # Create target points with a constant shift of 1 from pred_points
    target_points = pred_points.clone() + 1.0  # Add 1.0 to all coordinates
    print(f"pred_points: {pred_points.shape}, target_points: {target_points.shape}")
    
    # Test different loss types
    for loss_type in ['mse', 'chamfer',  'chamfer_pytorch3d', 'combined_pytorch3d', 'combined',]:
        print(f"\nTesting {loss_type} loss:")
        point_loss_fn = PointCloudLoss(exp_config, loss_type=loss_type)
        start_time = time.time()
        point_loss = point_loss_fn(pred_points, target_points)
        point_time = time.time() - start_time
        print(f"PointCloudLoss ({loss_type}): {point_loss.item():.4f}")
        print(f"Calculation time: {point_time:.4f} seconds")