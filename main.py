import cv2
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset,random_split
import yaml
import os
import argparse
from tqdm import tqdm
import torch
import math
import pickle
from torch.utils.tensorboard import SummaryWriter
import os
# import timm
# from timm.scheduler import CosineLRScheduler
# from timm.loss.cross_entropy import LabelSmoothingCrossEntropy
# from lion_pytorch import Lion
from ThermalDataset import *
from Models import *
from utils import *
from Losses import DepthMaskLoss, PointCloudLoss, CombinedLoss
from Metrics import DepthMaskMetrics, PointCloudMetrics, DepthMask2PointCloud, DepthMask2PointCloudFast
from output_check_visualize import OutputVisualizer
import sys
import traceback
import random


def get_gpu_ids(i):
    """Map a logical GPU index to a CUDA UUID via nvidia-smi.

    Useful on multi-GPU / MIG servers where UUID pinning is more reliable
    than integer indices. Returns None if nvidia-smi is unavailable.
    """
    try:
        gpu_info = os.popen("nvidia-smi -L").readlines()
    except Exception:
        return None
    if not gpu_info:
        return None
    gpu_ids = []
    for line in gpu_info:
        ids = line.split("UUID: ")[-1].strip(" ()\n")
        if ids.startswith("GPU"):
            continue
        gpu_ids.append(ids)
    if not gpu_ids or i < 0 or i >= len(gpu_ids):
        return None
    return str(gpu_ids[i])


def setup_device(cuda_index):
    """Select torch device. Prefer GPU if available; otherwise fall back to CPU.

    On NVIDIA machines, try UUID-based CUDA_VISIBLE_DEVICES pinning
    (lab / MIG friendly). If that fails, use the integer index instead.
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        return torch.device("cpu")

    gpu_uuid = get_gpu_ids(cuda_index)
    if gpu_uuid is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
        print(f"cuda index is: {cuda_index}")
        print(f"cuda visible devices (UUID): {gpu_uuid}")
    else:
        # Portable fallback for normal single-/multi-GPU machines
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_index)
        print(f"cuda index is: {cuda_index}")
        print(f"cuda visible devices (index): {cuda_index}")

    # After setting CUDA_VISIBLE_DEVICES, the selected GPU is always cuda:0
    return torch.device("cuda:0")

def get_config(config_folder, config_file_name):
    if '.yaml' not in config_file_name:
        config_file_name = config_file_name + '.yaml'

    all_yaml_files = []
    all_yaml_file_paths = []
    for root, dirs, files in os.walk(config_folder):
        for file in files:
            if file.endswith(".yaml"):
                all_yaml_files.append(file)
                all_yaml_file_paths.append(os.path.join(root, file))

    if config_file_name not in all_yaml_files:
        print("Configuration file name is: ",config_file_name)
        print("The configuration file is not found! Please check the file name!")
        exit
    else:
        # if there are multiple configuration files with the same name, print all the locations
        if all_yaml_files.count(config_file_name) > 1:
            print("The configuration file is: ",config_file_name)
            print("There are multiple configuration files with the same name!")
            for i in range(all_yaml_files.count(config_file_name)):
                print(all_yaml_file_paths[all_yaml_files.index(config_file_name, i)])
            exit
        else:
            config_file_path = all_yaml_file_paths[all_yaml_files.index(config_file_name)]
            print("The configuration file is found at: ",config_file_path)
            with open(config_file_path, 'r') as fd:
                config = yaml.load(fd, Loader=yaml.FullLoader)
    return config

def correct_none_in_config(config):
    # traverse the configuration file if any value is a string: 'None', convert it to None
    for key, value in config.items():
        if value == 'None':
            config[key] = None
    return config
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
if __name__ == "__main__":
    '''
    The main function for training and testing the model
    can be run with the following command:
        python main.py --exp_config_file 'initial_unet_exp' --cuda_index 0 --mode 0 --pretrained_model 'pretrained_model_weights' --vis_enable 1
    '''
    parser = argparse.ArgumentParser(description='Training and Testing')
    parser.add_argument("--exp_config_file", type=str, help="Configuration YAML file of the experiment")
    parser.add_argument("--cuda_index", type=int, default=0, help="The index of the cuda device")
    parser.add_argument("--mode", type=int, default=0, help="0: train + test, 1: test only, 2: finetune/resume training + test, 3: check the pipeline")
    parser.add_argument("--pretrained_model", type=str, default=None, help="The file path of the pretrained model weights")
    parser.add_argument("--vis_enable", type=int, default=0, help="enable visualization by directly output results to video file")
    parser.add_argument("--trainset_portion", type=float, default=1.0, help="the portion of the trainset to use for training")
    parser.add_argument("--pertureb_temperature", type=int, default=0, help="enable pertureb temperature for the testing (evaluating the sensitivity of the model), 0: no pertureb, otherwise: pertureb with the value (the value is the temperature value to pertureb).")
    parser.add_argument("--DIM_sensitivity_evaluation", type=int, default=0, help="evaluate the sensitivity of the DIM parameters, 0: no evaluation, 1: evaluation")
    parser.add_argument("--DIM_bin_size", type=float, default=20., help="the bin size (mm) of the DIM")
    parser.add_argument("--DIM_depth_max", type=float, default=8000., help="the depth max (mm) of the DIM")
    parser.add_argument("--DIM_sigma", type=float, default=0.5, help="the sigma of the DIM")
    args = parser.parse_args()
    
    # loading the configuration file ########################################
    exp_config_file_name = args.exp_config_file + '.yaml'
    exp_config = get_config('exp_configs', exp_config_file_name)
    exp_config = correct_none_in_config(exp_config)
    if args.mode == 1: # only need the testset configuration file
        trainset_config = get_config('data_configs', exp_config['trainset_config_file'])  # this is not used
        trainset_config = correct_none_in_config(trainset_config)   # this is not used
        testset_config = get_config('data_configs', exp_config['testset_config_file'])
        testset_config = correct_none_in_config(testset_config)
    else:
        trainset_config = get_config('data_configs', exp_config['trainset_config_file'])
        trainset_config = correct_none_in_config(trainset_config)
        testset_config = get_config('data_configs', exp_config['testset_config_file'])
        testset_config = correct_none_in_config(testset_config)
    
    exp_config['cuda_index'] = args.cuda_index
    
    
    if args.DIM_sensitivity_evaluation == 1: # this is for evaluating the sensitivity of the proposed TAP3D model to the DIM parameters, use the default DIM parameters if not provided
        exp_config['bin_size'] = args.DIM_bin_size
        exp_config['depth_max'] = args.DIM_depth_max
        exp_config['sigma'] = args.DIM_sigma
        print(f"DIM parameters are used for the sensitivity evaluation: bin_size: {args.DIM_bin_size}, depth_max: {args.DIM_depth_max}, sigma: {args.DIM_sigma}")
    else:
        print("DIM parameters are not used for the sensitivity evaluation! Use the default DIM parameters!")
    
    
    if args.mode == 1:
        print("Testing only!")
        if args.pretrained_model == None:
            print("Please provide the pretrained model weights!")
            exit
    elif args.mode == 2:
        print("Finetuning and Testing!")
        if args.pretrained_model == None:
            print("Please provide the pretrained model weights!")
            exit
    elif args.mode == 3:
        print("Check the pipeline!")
        exp_config['num_epochs'] = 1
    else:
        print("Training and Testing!")
        
    tensorboard_folder = exp_config['tensorboard_folder']  # the folder to save the tensorboard logs
    if not os.path.exists(tensorboard_folder):
        os.makedirs(tensorboard_folder)
    trained_model_folder = exp_config['trained_model_folder']  # the folder to save the trained model weights
    if not os.path.exists(trained_model_folder):
        os.makedirs(trained_model_folder)
    model_save_enable = exp_config.get('model_save_enable', True)  # by default, save the model weights
    
    device = setup_device(exp_config['cuda_index'])
    print("The device is: ", device)
    
    # fix the seed for reproducibility
    rng_generator = torch.manual_seed(exp_config['init_rand_seed'])
    if device.type == "cuda":
        torch.cuda.manual_seed(exp_config['init_rand_seed'])
    np.random.seed(exp_config['init_rand_seed'])
    random.seed(exp_config['init_rand_seed'])
    
    localtime = time.localtime(time.time())  # get the current time for the log file name
    if args.mode == 1:
        if args.trainset_portion < 1.0:
            trainset_portion_str = str(args.trainset_portion).replace('.', '_')
            log_file_name = f"{args.exp_config_file}_{exp_config['model_name']}_{time.strftime('%m%d%H%M%S', localtime)}_test_trainset_{trainset_portion_str}"
        else:
            log_file_name = f"{args.exp_config_file}_{exp_config['model_name']}_{time.strftime('%m%d%H%M%S', localtime)}_test"
    elif args.mode == 2:
        if args.trainset_portion < 1.0:
            trainset_portion_str = str(args.trainset_portion).replace('.', '_')
            log_file_name = f"{args.exp_config_file}_{exp_config['model_name']}_{time.strftime('%m%d%H%M%S', localtime)}_finetune_trainset_{trainset_portion_str}"
        else:
            log_file_name = f"{args.exp_config_file}_{exp_config['model_name']}_{time.strftime('%m%d%H%M%S', localtime)}_finetune"
    else:
        if args.trainset_portion < 1.0:
            trainset_portion_str = str(args.trainset_portion).replace('.', '_')
            log_file_name = f"{args.exp_config_file}_{exp_config['model_name']}_{time.strftime('%m%d%H%M%S', localtime)}_trainset_{trainset_portion_str}"
        else:
            log_file_name = f"{args.exp_config_file}_{exp_config['model_name']}_{time.strftime('%m%d%H%M%S', localtime)}"
    
    if args.pertureb_temperature != 0:
        log_file_name = f"{log_file_name}_pertureb_{args.pertureb_temperature}"
        exp_config['pertureb_temperature'] = args.pertureb_temperature
    else:
        exp_config['pertureb_temperature'] = None
        
    if args.DIM_sensitivity_evaluation == 1:
        log_file_name = f"{log_file_name}_DIM_sensitivity_evaluation_{str(args.DIM_bin_size).replace('.', '_')}_{str(args.DIM_depth_max).replace('.', '_')}_{str(args.DIM_sigma).replace('.', '_')}"
    else:
        pass
        
    print("The log filename is: ",log_file_name)
    
    log_folder = exp_config['log_folder']   # save all the outputs records when testing
    log_enable = exp_config['log_enable']   # if log_enable is False, the log folder will not be created
    log_folder = os.path.join(log_folder, log_file_name)
    if not os.path.exists(log_folder) and log_enable:
        os.makedirs(log_folder)    
        print(f"log folder is {log_folder}")
    video_path = None   # the path to save the video output (visualization)
    if args.vis_enable:
        video_path = os.path.join(log_folder, log_file_name + ".avi")
        print(f"video path is {video_path}")

    writer = SummaryWriter(exp_config['tensorboard_folder'] + log_file_name)
    
    # configure the model, optimizer, scheduler, criterion, and metric  ########################################
    model_name = exp_config['model_name']
    model = get_registered_models(model_name, exp_config)
    model.to(device)
    
    if args.mode == 1 or args.mode == 2:  # test-only or finetune-test: need to load the pretrained model weights
        # map_location is required for CPU-only machines loading GPU-trained checkpoints
        state_dict = torch.load(args.pretrained_model, map_location=device)
        if args.mode == 2:
            # for key in list(state_dict.keys()):
            #     if 'mlp_head' in key:
            #         del state_dict[key]
            #         print("deleted:", key)
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)
        print("The pretrained model weights are loaded!")
    
    if args.mode == 3: # for pipeline check, we only use partially of the data to check the pipeline
        trainset_config['sub_folder_list'] = trainset_config['sub_folder_list'][:2]
        testset_config['sub_folder_list'] = testset_config['sub_folder_list'][:2]
    else:
        if args.trainset_portion < 1.0:
            trainset_config['sub_folder_list'] = trainset_config['sub_folder_list'][:int(len(trainset_config['sub_folder_list'])*args.trainset_portion)]
        else:
            pass
    
    
    trainset = MyData(exp_config, trainset_config)
    train_num = int(len(trainset)*0.8)
    val_num = len(trainset) - train_num
    print(f"DEBUG: trainset class: array with {len(trainset)} elements; trainset 1st level: array with {len(trainset[0])} elements; trainset 2nd level: {trainset[0][0].shape}, what its [0] level looks like: {trainset[0][0][0].shape}")
    train_set, val_set = torch.utils.data.random_split(trainset, [train_num, val_num], generator=rng_generator)
    train_loader = DataLoader(train_set, batch_size=exp_config['batch_size'], shuffle=True, num_workers=exp_config['num_workers'], worker_init_fn=seed_worker)
    val_loader = DataLoader(val_set, batch_size=exp_config['batch_size'], shuffle=False, num_workers=exp_config['num_workers'], worker_init_fn=seed_worker)
    print("The train set is: ",len(train_set))
    print("The val set is: ",len(val_set))
        
    if exp_config['optimizer'] == "AdamW":
        momentum = exp_config.get('momentum', 0.9)
        optimizer = torch.optim.AdamW(model.parameters(), lr=exp_config['lr'], betas=(momentum, momentum + momentum/10), weight_decay=exp_config['weight_decay'])                      
    elif exp_config['optimizer'] == "Adam":
        momentum = exp_config.get('momentum', 0.9)
        optimizer = torch.optim.Adam(model.parameters(), lr=exp_config['lr'], betas=(momentum, momentum + momentum/10), weight_decay=exp_config['weight_decay'])
    else:
        momentum = exp_config.get('momentum', 0.9)
        optimizer = torch.optim.SGD(model.parameters(), lr=exp_config['lr'], momentum=momentum, weight_decay=exp_config['weight_decay'])
    warmup_steps = 10
    lr_func = lambda step: min((step + 1) / (warmup_steps + 1e-8), 0.5 * (math.cos(step / exp_config['num_epochs'] * math.pi) + 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)

    if exp_config['criterion'] == 'depthmask3C_loss':
        criterion = DepthMaskLoss(exp_config)
    elif exp_config['criterion'] == 'pointcloud_loss':
        criterion = PointCloudLoss(exp_config)
    elif exp_config['criterion'] == 'combined_loss':
        criterion = CombinedLoss(exp_config, debug=False)
    else:
        raise NotImplementedError
    criterion.to(device)

    if exp_config['metric'] == 'depthmask3C_metrics':
        metric = DepthMaskMetrics(exp_config)
    elif exp_config['metric'] == 'pointcloud_metrics':
        metric = PointCloudMetrics(exp_config)
    else:
        raise NotImplementedError
    if exp_config['depthmask_to_pointcloud']: # convert the depth mask to point cloud and then calculate the metric with point cloud metrics for better comparison
        depthmask2pointcloud = DepthMask2PointCloudFast(exp_config)
        depthmask2pointcloud.to(device)
        metric = PointCloudMetrics(exp_config)
    else:
        depthmask2pointcloud = None
    metric.to(device)
    
    # training/finetuning  ########################################
    best_model_epoch = 0
    if args.mode == 0:
        print("Start training!")
        best_model_weights, best_model_epoch = train(model,train_loader,val_loader,criterion,optimizer,scheduler,metric,exp_config,writer,device, depthmask2pointcloud)
        print("Training is done!")
        print("The best model is at epoch: ",best_model_epoch)
        model.load_state_dict(best_model_weights)
        if model_save_enable:
            saved_model_path = exp_config['trained_model_folder']+ log_file_name + '.pth'
            torch.save(best_model_weights, saved_model_path)
    elif args.mode == 1:
        print("Start testing!")
    elif args.mode == 2 or args.mode == 3:
        print("Start finetuning or checking!")
        best_model_weights, best_model_epoch = train(model,train_loader,val_loader,criterion,optimizer,scheduler,metric,exp_config,writer,device, depthmask2pointcloud)
        print("Finetuning or checking is done!")
        print("The best model is at epoch: ",best_model_epoch)
        model.load_state_dict(best_model_weights)
        if model_save_enable:
            saved_model_path = exp_config['trained_model_folder']+ log_file_name + '.pth'
            torch.save(best_model_weights, saved_model_path)
    else:
        print("The mode is wrong!")
        exit
    
    print("Start testing!")
    test_sub_folder_list = testset_config['sub_folder_list']
    metric_all_folders = []
    total_num_batch = 0
    total_loss = 0
    failed_test_sub_folders = []

    if args.vis_enable:
        output_visualizer = OutputVisualizer()
        output_visualizer.create_image_output_dest(video_path)
    for test_sub_folder in test_sub_folder_list:
        try:
            print(f"=========================start: {test_sub_folder}===========================")
            recordings_all_folders = {}
            # we will conducting the testing on sub_foder one by one
            testset_config['sub_folder_list'] = [test_sub_folder]
            test_set = MyData(exp_config, testset_config)
            print("The test sub folder is: ",test_sub_folder)
            print("The test set is: ",len(test_set))
            test_loader = DataLoader(test_set, batch_size=exp_config['batch_size'], shuffle=False, num_workers=exp_config['num_workers'], worker_init_fn=seed_worker)
            recordings, loss_all = inference(model,test_loader,exp_config,device, criterion, metric, depthmask2pointcloud)
            print(f'Average Loss (test set) {loss_all/ len(test_loader):.10f}')
            total_num_batch += len(test_loader)
            total_loss += loss_all
            if log_enable or args.vis_enable:
                recordings_all_folders[test_sub_folder] = recordings
                # recordings_all_folders[test_sub_folder]['exp_config'] = exp_config
            metric_all = recordings['metrics']
            metric_all_folders.append(metric_all)
            if len(metric_all) > 0:
                for key in metric_all[0].keys():
                    metric_list = [metric[key] for metric in metric_all]
                    metric_mean = np.mean(np.array(metric_list))
                    writer.add_scalar(f'test {key} (per sample)', metric_mean, best_model_epoch)
            
            if args.vis_enable:
                output_visualizer.load_data_direct(recordings_all_folders[test_sub_folder], test_sub_folder)
                output_visualizer.loaded_data_to_video_overlap()
            
            # dump data to pickle files
            if log_enable:
                test_result_save_path = os.path.join(log_folder, test_sub_folder + ".pkl")
                recordings_all_folders['exp_config'] = exp_config
                recordings_all_folders['failed_test_sub_folders'] = failed_test_sub_folders
                pickle.dump(recordings_all_folders, open(test_result_save_path, 'wb'))
                print("The test results are saved at: ",test_result_save_path)
            recordings_all_folders = {}
    
        except Exception as e:
            failed_test_sub_folders.append(test_sub_folder)
            print(f"Failed to test {test_sub_folder}: error message: {e}")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print("An error occurred:")
            traceback.print_exception(exc_type, exc_value, exc_traceback)

    # uncomment below two lines to convert whole
    if args.vis_enable:
        output_visualizer.close_image_output_dest()
    print(f'Average Loss (test set) {total_loss/ total_num_batch:.10f}')
    writer.add_scalar('test loss (per sample)', total_loss/ total_num_batch, best_model_epoch)
    
    metric_keys = list(metric_all_folders[0][0].keys())
    metric_mean_all = {}
    for key in metric_keys:
        metric_mean_all[key] = []
    for i in range(len(metric_all_folders)):
        metric_all = metric_all_folders[i]
        if len(metric_all) > 0:
            for key in metric_keys:
                metric_list = [metric[key] for metric in metric_all]
                metric_mean = np.mean(np.array(metric_list))
                metric_mean_all[key].append(metric_mean)
    for key in metric_keys:
        mean_metric = np.mean(np.array(metric_mean_all[key]))
        writer.add_scalar(f'test {key} (per sample)', mean_metric, best_model_epoch)
    writer.close()
