from .UNetLike import UNetLike, UNetLike2
from .ThermoPT import ThermoPT
from .newcrfs import NewCRFDepth
from .rgb2point import PointCloudNet


registered_models= {
    'unet_like': UNetLike2,
    'thermo_pt': ThermoPT,
    'newcrf_depth': NewCRFDepth,
    'rgb2point': PointCloudNet,
}

def get_registered_models(model_name, config):
    if 'unet_like' in model_name:
        in_channels = config['chunk_size']
        return registered_models[model_name](in_channels = in_channels, out_channels = config['out_channels'], init_features = config['init_features'])
    elif 'thermo_pt' in model_name:
        enable_aov_refine = config.get('enable_aov_refine', True)
        enable_bev_refine = config.get('enable_bev_refine', True)
        enable_reconstruction = config.get('enable_reconstruction', True)
        enable_multi_primitive = config.get('enable_multi_primitive', True)
        return registered_models[model_name](exp_config = config, in_channels = config['chunk_size'], out_channels = config['out_channels'], init_features = config['init_features'], enable_aov_refine = enable_aov_refine, enable_bev_refine = enable_bev_refine, enable_reconstruction = enable_reconstruction, enable_multi_primitive = enable_multi_primitive)
    elif 'newcrf_depth' in model_name:
        return registered_models[model_name](version='tiny07', inv_depth=False, max_depth=config['max_depth'], pretrained=None)
    elif 'rgb2point' in model_name:
        return registered_models[model_name](num_views=config['num_views'], num_heads=config['num_heads'], dim_feedforward=config['dim_feedforward'], exp_config=config)
    else:
        print("The model name is not registered!")
        return None
