import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .UNetLike import UNetLike2


def print_param_grad_table(model):
    print(f"{'param':60s} | {'||grad||':>10s} | {'|grad|_max':>10s} | {'finite':>6s}")
    print("-"*100)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            print(f"{name:60s} | {'None':>10s} | {'':>10s} | {'':>6s}")
        else:
            g = p.grad
            finite = "Y" if torch.isfinite(g).all() else "N"
            print(f"{name:60s} | {g.norm().item():10.3e} | {g.abs().max().item():10.3e} | {finite:>6s}")

torch.autograd.set_detect_anomaly(True)


def depth_maps_to_point_cloud_torch(depth_maps, hfov_deg, vfov_deg):
    """
    depth_maps: torch.Tensor, (B, H, W)
    hfov_deg, vfov_deg: float or with shape (B,) tensor/list/ndarray 
    reture: torch.Tensor, (B, H, W, 3)
    """
    if depth_maps.ndim != 3:
        raise ValueError("depth_maps must be a  (B, H, W) tensor.")
    device = depth_maps.device
    dtype = depth_maps.dtype
    B, H, W = depth_maps.shape

    hfov_rad = torch.deg2rad(torch.as_tensor(hfov_deg, device=device, dtype=dtype))
    vfov_rad = torch.deg2rad(torch.as_tensor(vfov_deg, device=device, dtype=dtype))
    fx = W / (2.0 * torch.tan(hfov_rad / 2.0))  # scalar or (B,)
    fy = H / (2.0 * torch.tan(vfov_rad / 2.0))  # scalar or (B,)

    if fx.ndim == 1:
        if fx.numel() != B:
            raise ValueError("hfov_deg must be B.")
        fx = fx.view(B, 1, 1)
    if fy.ndim == 1:
        if fy.numel() != B:
            raise ValueError("vfov_deg must be B.")
        fy = fy.view(B, 1, 1)

    xs = torch.arange(W, device=device, dtype=dtype)
    ys = torch.arange(H, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')        # yy: row(y), xx: column(x), shape (H, W)
    xx = xx.unsqueeze(0) - W / 2.0                        # (1, H, W)
    yy = yy.unsqueeze(0) - H / 2.0                        # (1, H, W)

    x_cam = xx / fx                                       # (B, H, W)
    y_cam = yy / fy                                       # (B, H, W)
    z_cam = depth_maps

    x_3d = x_cam * z_cam
    y_3d = y_cam * z_cam

    pc = torch.stack([x_3d, y_3d, z_cam], dim=-1)         # (B, H, W, 3)
    return pc


class LearnableConsts(nn.Module):
    def __init__(self):
        super().__init__()
        # raw, unconstrained parameters
        self.c1 = nn.Parameter(torch.log(torch.exp(torch.tensor(1.0))-1))  # inverse softplus init
        self.c2 = nn.Parameter(torch.log(torch.exp(torch.tensor(1.0))-1))
        self.c3 = nn.Parameter(torch.log(torch.exp(torch.tensor(1.0))-1))
        self.c4 = nn.Parameter(torch.log(torch.exp(torch.tensor(1.0))-1))
        self.gain = nn.Parameter(torch.log(torch.exp(torch.tensor(1.0))-1))
        self.D = nn.Parameter(torch.log(torch.exp(torch.tensor(0.01))-1))
        self.f = nn.Parameter(torch.log(torch.exp(torch.tensor(0.01))-1))
        self.zeta = nn.Parameter(torch.log(torch.exp(torch.tensor(1.0))-1))
        self.gamma = nn.Parameter(torch.log(torch.exp(torch.tensor(1e-3))-1))
        self.xi_raw = nn.Parameter(torch.tensor(0.0))  # sigmoid later

    def forward(self):
        # positive-only with softplus
        C1 = F.softplus(self.c1);  C2 = F.softplus(self.c2)
        C3 = F.softplus(self.c3);  C4 = F.softplus(self.c4)
        G  = F.softplus(self.gain)
        D  = F.softplus(self.D)
        f  = F.softplus(self.f)
        z  = F.softplus(self.zeta)
        g  = F.softplus(self.gamma)
        xi = torch.sigmoid(self.xi_raw)
        return C1,C2,C3,C4,G,D,f,z,g,xi
    
    

class LearnableConstsTensor(nn.Module):
    def __init__(self, H: int, W: int):
        super().__init__()
        # Use inverse softplus initialization for better gradient flow
        inv_softplus_1 = torch.log(torch.exp(torch.tensor(1.0)) - 1)
        inv_softplus_small = torch.log(torch.exp(torch.tensor(0.01)) - 1)
        
        # --- Per-pixel maps (proper initialization) ---
        self.c1    = nn.Parameter(torch.full((H, W), inv_softplus_1))
        self.c2    = nn.Parameter(torch.full((H, W), inv_softplus_1))
        self.c3    = nn.Parameter(torch.full((H, W), inv_softplus_1))
        self.c4    = nn.Parameter(torch.full((H, W), inv_softplus_1))
        self.gain  = nn.Parameter(torch.full((H, W), inv_softplus_1))
        self.zeta  = nn.Parameter(torch.full((H, W), inv_softplus_1))
        self.xi_raw = nn.Parameter(torch.zeros(H, W))   # sigmoid -> 0.5

        # --- Global scalars (proper initialization) ---
        self.D     = nn.Parameter(torch.full((), inv_softplus_small))      # scalar
        self.f     = nn.Parameter(torch.full((), inv_softplus_small))      # scalar
        self.gamma = nn.Parameter(torch.full((), inv_softplus_small))      # scalar

    def forward(self):
        # Positive-only with softplus; xi in (0,1) with sigmoid
        C1 = F.softplus(self.c1);  C2 = F.softplus(self.c2)
        C3 = F.softplus(self.c3);  C4 = F.softplus(self.c4)
        G  = F.softplus(self.gain)
        D  = F.softplus(self.D)
        f  = F.softplus(self.f)
        z  = F.softplus(self.zeta)
        g  = F.softplus(self.gamma)
        xi = torch.sigmoid(self.xi_raw)
        return C1, C2, C3, C4, G, D, f, z, g, xi



class ThermoPT(nn.Module):
    def __init__(self, 
                 exp_config, 
                 in_channels=1,     # the input is single channel temperature map from thermal array sensor
                 out_channels=5,    # the output is 5-channels: [Depth, User_index_map, Emissivity, Reflectance, Temperature]        
                 init_features=32,   
                 enable_aov_refine = True,   # whether to enable the AOV refinement
                 enable_bev_refine = True,   # whether to enable the BEV refinement
                 enable_reconstruction = True, # whether to enable the reconstruction
                 enable_multi_primitive = True, # whether to enable the multi-primitive (otherwise, only Depth and User_index_map are predicted)
                 ):    
        super(ThermoPT, self).__init__()
        self.enable_aov_refine = enable_aov_refine
        self.enable_bev_refine = enable_bev_refine
        self.enable_reconstruction = enable_reconstruction
        self.enable_multi_primitive = enable_multi_primitive
        self.ssl_enable = exp_config.get('ssl_enable', False) # whether to enable the Self-Supervised Learning (SSL) training
        print(f"enable_aov_refine: {enable_aov_refine}, enable_bev_refine: {enable_bev_refine}, enable_reconstruction: {enable_reconstruction}, enable_multi_primitive: {enable_multi_primitive}")
        
        if enable_aov_refine:
            self.aov_refine = UNetLike2(3, 1, init_features)  # input the maps: Depth, Emissivity, Temperature, output the user mask
        if enable_bev_refine:
            self.bin_size = exp_config.get('bin_size', 20.)
            self.depth_max = exp_config.get('depth_max', 8000.)
            self.sigma = exp_config.get('sigma', .5)
            self.bev_refine = UNetLike2(1, 1, init_features)  # input the BEV histogram, output the refined BEV histogram
        if enable_reconstruction:   # if enable the reconstruction, we will predict the Depth, User_index_map, Emissivity, Reflectance, Temperature
            self.sensor_name = exp_config['sensor_name']
            if self.sensor_name == 'senxor_m08':
                self.hfov_deg, self.vfov_deg = 90, 67
                self.height, self.width = 62, 80
            elif self.sensor_name == 'senxor_m16':
                self.hfov_deg, self.vfov_deg = 88.5, 58.5
                self.height, self.width = 120, 160
            elif self.sensor_name == 'seek_thermal':
                self.hfov_deg, self.vfov_deg = 81, 59
                self.height, self.width = 150, 200
            # self.consts = LearnableConsts()
            self.consts = LearnableConstsTensor(self.height, self.width)
            # Stefan-Boltzmann constant, used for ∫M_bb ≈ σ T^4 approximation
            self.sigma_sb = 5.670374419e-8
            self.inverse_solver = UNetLike2(in_channels, out_channels, init_features)  # the inverse solver to predict the Depth, User_index_map, Emissivity, Reflectance, Temperature
        else:
            if self.enable_multi_primitive:
                self.inverse_solver = UNetLike2(in_channels, out_channels, init_features)  # the inverse solver to predict the Depth, User_index_map, Emissivity, Reflectance, Temperature
            else:
                out_channels = 2 # Depth, User_index_map
                self.inverse_solver = UNetLike2(in_channels, out_channels, init_features)  # the inverse solver to predict the Depth, User_index_map

    def forward(self, x):
        if not self.enable_reconstruction:
            if self.enable_multi_primitive:
                out = self.inverse_solver(x)
                Depth_raw      = out[:, 0:1]       # (B,1,H,W)
                Depth      = F.softplus(Depth_raw) + 1e-6               # >0, shape (B,1,H,W)
                User_index_raw = out[:, 1:2]
                return {
                    "raw": out,                 # (B,2,H,W): [Depth, User_index_map]
                    "user_index": User_index_raw,  # (B,1,H,W)
                    'depth': Depth,             # (B,1,H,W)
                }
            else:
                out = self.inverse_solver(x)
                Depth_raw      = out[:, 0:1]       # (B,1,H,W)
                Depth      = F.softplus(Depth_raw) + 1e-6               # >0, shape (B,1,H,W)
                User_index_raw = out[:, 1:2]
                return {
                    "raw": out,                 # (B,2,H,W): [Depth, User_index_map]
                    "user_index": User_index_raw,  # (B,1,H,W)
                    'depth': Depth,             # (B,1,H,W)
                }
        else:
            final_return = {}
            out = self.inverse_solver(x)
            Depth_raw      = out[:, 0:1]       # (B,1,H,W)
            User_index_raw = out[:, 1:2]
            Emissivity_raw = out[:, 2:3]
            Reflect_raw    = out[:, 3:4]
            Temp_raw   = out[:, 4:5]
            
            final_return['raw'] = out # (B,5,H,W): [Depth, User_index_map, Emissivity, Reflectance, Temperature]
            final_return['user_index'] = User_index_raw # (B,1,H,W)
            # learnable constants (positive-only with softplus)
            C1, C2, C3, C4, G, D, f, zeta, gamma, xi = self.consts()            
            # physical range constraint (keep differentiable)
            eps = 1e-8
            Depth = F.softplus(Depth_raw) + 1e-6               # >0, shape (B,1,H,W)
            Emissivity = torch.sigmoid(Emissivity_raw)              # (0,1)
            Reflectance= F.softplus(Reflect_raw)                    # >=0 (aggregated ambient reflection "energy" term)
            Temperature= F.softplus(Temp_raw) + 1e-6                # here we assume the network output is in Kelvin; if in °C, please add 273.15

            final_return['depth'] = Depth # (B,1,H,W)
            final_return['Emissivity'] = Emissivity # (B,1,H,W)
            final_return['Reflectance'] = Reflectance # (B,1,H,W)
            final_return['Temperature'] = Temperature # (B,1,H,W)

            # 3D position of pixels (B,H,W,3), z is the optical axis direction, ie, the depth axis
            if self.ssl_enable:
                xyz = depth_maps_to_point_cloud_torch(Depth.squeeze(1),
                                                hfov_deg=self.hfov_deg,
                                                vfov_deg=self.vfov_deg)  # (B,H,W,3)
            else:
                xyz = depth_maps_to_point_cloud_torch(Depth.detach().squeeze(1),
                                                    hfov_deg=self.hfov_deg,
                                                    vfov_deg=self.vfov_deg)  # (B,H,W,3)
            x3d = xyz[..., 0]
            y3d = xyz[..., 1]
            z3d = xyz[..., 2]                          # = depth
            r   = torch.sqrt(x3d*x3d + y3d*y3d + z3d*z3d + eps)  # distance ||P_t||

            # angle term: <P_t, x>=z, cos α = z/r
            cos4 = (z3d**4) / (r**4 + eps)    # <p_t, x>^{4} / ||P_t||^{4} in Equation (7)
            cos7 = (z3d**7) / (r**5 + eps)      # <p_t, x>^{7} / ||P_t||^{5} in Equation (8)

            # atmospheric attenuation
            atten = torch.exp(-gamma * r)              # e^{-γ||P_t||} in Equation (7,8)

            # blackbody integral approximation: ∫ M_bb(T) dλ ≈ σ T^4
            Mbb = self.sigma_sb * (Temperature.squeeze(1) ** 4)    # (B,H,W) in Equation (7)

            # V1: spontaneous radiation: equation (7)
            k1 = (G * (D**2) * zeta) / (f**2 + eps)
            V1 = k1 * cos4 * atten * Emissivity.squeeze(1) * Mbb   # (B,H,W)

            # V2: reflected radiation equation (8), use predicted Reflectance to replace the summation term
            # here we do not explicitly use A_p, absorbed into the coefficients (optional to add A_p as a learnable constant)
            k2 = (G * (D**2) * zeta * xi) / (f**4 + eps)
            V2 = k2 * cos7 * atten * (1.0 - Emissivity.squeeze(1)) * Reflectance.squeeze(1)

            # composite voltage (ensure strictly positive to avoid div/log issues)
            V = (V1 + V2).clamp_min(1e-12)
            # Sakuma–Hattori: from voltage to temperature T_P (physically consistent) equation (9)
            # T_P = C2 / ( C3 * ln(C1 / V + 1) ) + C4 / C3
            log_arg = (C1 / V + 1.0).clamp_min(1.0 + 1e-12)
            log_val = torch.log(log_arg).clamp_min(1e-6)
            C3_safe = C3.clamp_min(1e-6)
            Tp = C2 / (C3_safe * log_val) + (C4 / C3_safe)  # (B,H,W)  # reconstructed temperature map
            
            final_return['Tp_recon'] = Tp.unsqueeze(1) # (B,1,H,W)
            final_return['V1'] = V1.unsqueeze(1) # (B,1,H,W)
            final_return['V2'] = V2.unsqueeze(1) # (B,1,H,W)
            
            # Multi-view refinement module:
            if self.enable_aov_refine:
                aov_input = torch.cat([Depth, Emissivity, Temperature], dim=1)  # (B,3,H,W)
                aov_output = self.aov_refine(aov_input)  # (B,1,H,W)
                final_return['aov_output'] = aov_output # (B,1,H,W)
            if self.enable_bev_refine:
                bev_input = DIM_backprop(Depth.squeeze(1), depth_max=self.depth_max, bin_size=self.bin_size, sigma=self.sigma, normalize=True, chunk_h=None)  # (B, W,D=401)
                # If the last dimension of bev_input is not 401 (align to our default DIM setting), interpolate it along that dimension with 'nearest' mode
                if bev_input.shape[-1] != 401:
                    bev_input = torch.nn.functional.interpolate(
                        bev_input.unsqueeze(1),  # (B, 1, W, D)
                        size=(bev_input.shape[-2], 401),
                        mode='nearest'
                    ).squeeze(1)  # (B, W, 401)
         
                bev_output = self.bev_refine(bev_input.unsqueeze(1))  # (B,1,W,D=401)
                final_return['bev_output'] = bev_output # (B,1,W,D=401)
                
            # merge the results:
            if self.enable_aov_refine and self.enable_bev_refine:
                human_mask = aov_output.detach().squeeze(1)  # (B,H,W)
                # make the mask to be zero-one mask
                human_mask = (human_mask > 0.5).float()
                bev_output = bev_output.detach().squeeze(1)  # (B,W,D=401)
                refined_depth = refine_depth_with_bev_bounds(Depth.detach().squeeze(1), bev_output, human_mask, bin_size=self.bin_size, threshold=0.8)[0] # returned shape [B,H,W]
                final_return['refined_depth'] = refined_depth.unsqueeze(1) # (B,1,H,W)
            elif self.enable_aov_refine:
                human_mask = aov_output.detach().squeeze(1)  # (B,H,W)
                # make the mask to be zero-one mask
                human_mask = (human_mask > 0.5).float()
                refined_depth = Depth.detach().squeeze(1) * human_mask  # shape (B,H,W)
                final_return['refined_depth'] = refined_depth.unsqueeze(1) # (B,1,H,W)
            elif self.enable_bev_refine:
                human_mask = None
                bev_output = bev_output.detach().squeeze(1)  # (B,W,D=401)
                refined_depth = refine_depth_with_bev_bounds(Depth.detach().squeeze(1), bev_output, human_mask, bin_size=self.bin_size, threshold=0.8)[0] # returned shape [B,H,W]
                final_return['refined_depth'] = refined_depth.unsqueeze(1) # (B,1,H,W)

            return final_return

    def check_consts_gradients(self):
        """Debug method to check gradients of learnable constants"""
        print("=== Learnable Constants Gradients ===")
        for name, param in self.consts.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_max = param.grad.abs().max().item()
                param_shape = list(param.shape)
                print(f"{name:15s} | Shape: {param_shape} | Grad Norm: {grad_norm:.6e} | Grad Max: {grad_max:.6e}")
            else:
                param_shape = list(param.shape)
                print(f"{name:15s} | Shape: {param_shape} | Grad: None")
        print("=" * 50)

    def check_consts_values(self):
        """Debug method to check actual values of learnable constants after activation functions"""
        with torch.no_grad():
            C1, C2, C3, C4, G, D, f, zeta, gamma, xi = self.consts()
            print("=== Learnable Constants Values (after activation) ===")
            print(f"C1:      min={C1.min().item():.6f}, max={C1.max().item():.6f}, mean={C1.mean().item():.6f}")
            print(f"C2:      min={C2.min().item():.6f}, max={C2.max().item():.6f}, mean={C2.mean().item():.6f}")
            print(f"C3:      min={C3.min().item():.6f}, max={C3.max().item():.6f}, mean={C3.mean().item():.6f}")
            print(f"C4:      min={C4.min().item():.6f}, max={C4.max().item():.6f}, mean={C4.mean().item():.6f}")
            print(f"G:       min={G.min().item():.6f}, max={G.max().item():.6f}, mean={G.mean().item():.6f}")
            print(f"D:       value={D.item():.6f}")
            print(f"f:       value={f.item():.6f}")
            print(f"zeta:    min={zeta.min().item():.6f}, max={zeta.max().item():.6f}, mean={zeta.mean().item():.6f}")
            print(f"gamma:   value={gamma.item():.6f}")
            print(f"xi:      min={xi.min().item():.6f}, max={xi.max().item():.6f}, mean={xi.mean().item():.6f}")
        print("=" * 50)


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



def refine_depth_with_bev_bounds(
    depth,      # (B, H, W), depth units must match bin_size (e.g., mm)
    bev,        # (B, W, D), binary or [0,1] occupancy from DIM
    mask_oav,   # (B, H, W), 1/True where valid (person), else 0/False
    bin_size,   # same bin_size used to build BEV
    threshold,   # treat bev > threshold as occupied
    ):
    """
    Implements:
        D_min(i) = min{ j | O_BEV(i,j) = 1 }
        D_max(i) = max{ j | O_BEV(i,j) = 1 }
        D*_t(k,i) = Clip(D_t(k,i), D_min(i)*bin_size, D_max(i)*bin_size) if M_OAV(k,i)=1 else 0

    Returns:
        depth_refined: (B, H, W)
        dmin_units:    (B, W)  (in depth units)
        dmax_units:    (B, W)  (in depth units)
    """
    if mask_oav is None:
        pass
    else:
        assert depth.ndim == 3 and bev.ndim == 3 and mask_oav.ndim == 3, "Shape mismatch"
    B, H, W = depth.shape
    Bb, Wb, D = bev.shape
    assert B == Bb and W == Wb, "Width/Batch mismatch between OAV depth and BEV"

    device = depth.device
    dtype = depth.dtype

    # Occupancy as boolean
    # bev go through sigmoid
    bev = torch.sigmoid(bev)
    occ = (bev > threshold)  # (B, W, D)

    # Indices along depth bins
    j = torch.arange(D, device=device, dtype=torch.long).view(1, 1, D)  # (1,1,D)

    # For min index: replace False with a large sentinel (D), then take min
    min_idx = torch.where(occ, j, torch.full_like(j, D)).min(dim=2).values  # (B, W)
    # For max index: replace False with 0, then take max
    max_idx = torch.where(occ, j, torch.zeros_like(j)).max(dim=2).values    # (B, W)

    # Handle columns with no occupancy at all: occ.any(dim=2)==False
    has_occ = occ.any(dim=2)  # (B, W)
    # If no occupied bin, make the clamp a no-op by spanning full range [0, (D-1)]
    min_idx = torch.where(has_occ, min_idx, torch.zeros_like(min_idx))
    max_idx = torch.where(has_occ, max_idx, (torch.full_like(max_idx, D - 1)))

    # Convert indices to depth units
    dmin_units = min_idx.to(dtype) * float(bin_size)  # (B, W)
    dmax_units = max_idx.to(dtype) * float(bin_size)  # (B, W)

    # Broadcast to (B, H, W)
    dmin_b = dmin_units[:, None, :]  # (B,1,W)
    dmax_b = dmax_units[:, None, :]  # (B,1,W)

    # Clamp each column (vectorized)
    depth_clamped = torch.clamp(depth, min=dmin_b, max=dmax_b)  # (B,H,W)

    if mask_oav is None: # no OAV mask, just return the clamped depth
        depth_refined = depth_clamped
    else: # Apply OAV mask: keep clamped where mask==1, else 0
        # mask_oav go through sigmoid
        mask_oav = torch.sigmoid(mask_oav)
        if mask_oav.dtype != torch.bool:
            mask_bool = mask_oav > 0.5
        else:
            mask_bool = mask_oav
        depth_refined = torch.where(mask_bool, depth_clamped, torch.zeros_like(depth_clamped))

    return depth_refined, dmin_units, dmax_units



if __name__ == "__main__":
    # test ThermoPT model
    exp_config = {
        'sensor_name': 'senxor_m08',
    }
    model = ThermoPT(exp_config, in_channels=1, out_channels=5, init_features=32, enable_aov_refine=True, enable_bev_refine=True, enable_reconstruction=True)
    
    x = torch.randn(2, 1, 62, 80)  # 假设输入形状 (B, C, H, W)
    output = model(x)
    
    for k in output.keys():
        print(k, output[k].shape)