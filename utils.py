import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

# excute the model and get the predictions and the labels
def inference(model,data_loader,config,device, criterion =None, metric = None, depthmask2pointcloud = None):
    model.eval()  # Set the model to evaluation mode
    recordings = {
        'predicts': [],
        'attributes': [],
        'labels': [],
        'metrics': [],
    }
    loss_all = 0
    with torch.no_grad():  # Disable gradient computation during evaluation
        for i, sample in enumerate(tqdm(data_loader, disable= config['tqdm_disable'])):
            input, label, *attr = sample
            recordings['attributes'].append(attr)
            input = input.float().to(device)   
            if config['pertureb_temperature'] is not None:
                input[0] = input[0] + torch.ones_like(input[0]) * config['pertureb_temperature']
            label = label.float().to(device)
            predict = model(input)
            
            if config['criterion'] == 'pointcloud_loss':
                # if the predict is in shape (batch_size, ...), need to reshape to (batch_size, 3, (max_num_points + 1 (indication)) * max_num_persons) to match the label
                if predict.ndim == 2:
                    predict = predict.reshape(predict.shape[0], 3, -1) 
            
            if criterion is not None: # if the loss function is not None, then calculate the loss
                if config['criterion'] == 'combined_loss':
                    loss = criterion(predict, label, input)
                else:
                    loss = criterion(predict, label)
                loss_all += loss.item()
            if metric is not None:
                if config['model_name'] == 'thermo_pt':
                    if 'refined_depth' in predict.keys():
                        depth = predict['refined_depth']
                        indicator = predict['user_index']
                        forground_background_mask = (indicator > 0.5).float()
                        predict = torch.cat([depth, indicator, forground_background_mask], dim=1)
                    else:
                        depth = predict['depth']
                        indicator = predict['user_index']
                        forground_background_mask = (indicator > 0.5).float()
                        predict = torch.cat([depth, indicator, forground_background_mask], dim=1)
                if depthmask2pointcloud is not None:
                    predict = depthmask2pointcloud(predict)
                    label = depthmask2pointcloud(label)
                metric_values = metric(predict, label)
                recordings['metrics'].append(metric_values)
            recordings['labels'].append(label.cpu().numpy())
            recordings['predicts'].append(predict.cpu().numpy())
    return recordings, loss_all


# model training
def train(model,train_loader,validation_loader,criterion,optimizer,scheduler, metric,config,writer,device, depthmask2pointcloud = None):
    best_loss = float('inf')
    best_model_weights = None
    best_model_epoch = 0
    max_num_persons = config['max_num_persons']
    max_num_points = config['max_num_points']

    for epoch in range(config['num_epochs']):
        model.train()  # Set the model to training mode
        loss_epoch = 0
        metric_dict_epoch = {}
        for i, sample in enumerate(tqdm(train_loader ,disable= config['tqdm_disable'])):
            input, label, *attr = sample
            loss = 0
            # with torch.autograd.detect_anomaly():
            input = input.float().to(device) 
            #print(f"DEBUG: shape of input is: {input.shape()}")
            #input[0] = cv2.resize(input[0], (224, 160), interpolation=cv2.INTER_LINEAR)
            label = label.float().to(device)
            predict = model(input)
            
            if config['criterion'] == 'pointcloud_loss':
                # if the predict is in shape (batch_size, ...), need to reshape to (batch_size, 3, (max_num_points + 1 (indication)) * max_num_persons) to match the label
                if predict.ndim == 2:
                    predict = predict.reshape(predict.shape[0], 3, -1) 
            
            #print(">>>>>>Pred shape:", predict.shape)
            #print(">>>>>>Label shape:", label.shape)
            if config['criterion'] == 'combined_loss':
                loss = criterion(predict, label, input)
            else:
                loss = criterion(predict, label)
            loss_epoch += loss.item()
            optimizer.zero_grad()
            loss.backward()
            # clip the gradient
            max_norm = config.get('max_norm', None)
            try:
                max_norm = config['max_norm']
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm, norm_type=2)
            except:
                pass
            
            optimizer.step()
            writer.add_scalar('train loss (per batch)', loss.item(), epoch * len(train_loader) + i)
            if metric is not None:
                if config['model_name'] == 'thermo_pt':
                    if 'refined_depth' in predict.keys():
                        depth = predict['refined_depth']
                        indicator = predict['user_index']
                        forground_background_mask = (indicator > 0.5).float()
                        predict = torch.cat([depth, indicator, forground_background_mask], dim=1)
                    else:
                        depth = predict['depth']
                        indicator = predict['user_index']
                        forground_background_mask = (indicator > 0.5).float()
                        predict = torch.cat([depth, indicator, forground_background_mask], dim=1)
                if depthmask2pointcloud is not None:
                    predict = depthmask2pointcloud(predict)
                    label = depthmask2pointcloud(label)
                metric_dict = metric(predict, label)
                for key in metric_dict.keys():
                    if key not in metric_dict_epoch.keys():
                        metric_dict_epoch[key] = []
                    metric_dict_epoch[key].append(metric_dict[key])
                    writer.add_scalar(f'train {key} (per batch)', metric_dict[key], epoch * len(train_loader) + i)
        if scheduler is not None:
            scheduler.step()
        # print(f'Epoch {epoch}, Average Loss (train set) {loss_epoch/ len(train_loader):.10f}')
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('train loss (per sample)', loss_epoch/ len(train_loader), epoch)
        if metric is not None:
            for key in metric_dict_epoch.keys():
                metric_epoch = np.mean(metric_dict_epoch[key])
                writer.add_scalar(f'train {key} (per sample)', metric_epoch, epoch)
                # print(f'Epoch {epoch}, Average {key} (train set) {metric_epoch:.10f}')
        
        recordings, loss_all = inference(model,validation_loader,config,device, criterion, metric, depthmask2pointcloud)
        metric_all = recordings['metrics']
        if len(metric_all) > 0:
            for key in metric_all[0].keys():
                metric_list = [metric[key] for metric in metric_all]
                metric_mean = np.mean(np.array(metric_list))
                writer.add_scalar(f'validation {key} (per sample)', metric_mean, epoch)
        writer.add_scalar('validation loss (per sample)', loss_all/ len(validation_loader), epoch)
        print(f'Epoch {epoch}, Average Loss (valid set) {loss_all/ len(validation_loader):.10f}')

        if loss_all<best_loss and (not np.isnan(loss_all)):
            best_model_weights = model.state_dict()
            best_loss = loss_all
            best_model_epoch = epoch
        else:
            if epoch - best_model_epoch >= config['early_stop']:    
                print('early stop with best model at epoch: ',best_model_epoch)
                break
        # if the loss is nan, then stop the training
        # if np.isnan(loss_all):
        #     print('early nan stop with best model at epoch: ',best_model_epoch)
        #     break
    return best_model_weights, best_model_epoch



def raw_data_loader(raw_data_folder, selected_recording_folder, slected_index_list, file_name_list_flag =False):
    """
    Load the raw data from the selected recording folder
    
    Parameters
    ----------
    raw_data_folder : str
        The path to the raw data folder
    selected_recording_folder : str
        The path to the selected recording folder
    slected_index_list : list
        The list of the selected index
    file_name_list_flag : bool
        The flag to return the file name list; the sensor name should be provided, 'seek_thermal', 'MLX', 'senxor_m08', 'senxor_m16'
    """
        
    MLX_data_folder = os.path.join(raw_data_folder, selected_recording_folder, 'MLX')
    realsense_depth_folder = os.path.join(raw_data_folder, selected_recording_folder, 'realsense_depth')
    realsense_color_folder = os.path.join(raw_data_folder, selected_recording_folder, 'realsense_color')
    seek_thermal_folder = os.path.join(raw_data_folder, selected_recording_folder, 'seek_thermal')
    senxor_m08_folder = os.path.join(raw_data_folder, selected_recording_folder, 'senxor_m08')
    senxor_m16_folder = os.path.join(raw_data_folder, selected_recording_folder, 'senxor_m16')
    
     # counting the files in real sense depth folder
    realsense_depth_files = os.listdir(realsense_depth_folder)
    # sort
    realsense_depth_files.sort()
    number_of_files = len(realsense_depth_files)
    # print('Number of samples:', number_of_files)
    
    MLX_data = []
    realsense_depth_data = []
    realsense_color_data = []
    seek_camera_data = []
    senxor_m08_data = []
    senxor_m16_data = []
    file_name_list = []
    
    if len(slected_index_list) == 0:
        # return all samples
        slected_index_list = range(number_of_files) 
        
    for slected_index in slected_index_list:
        if slected_index >= number_of_files:
            print('Index out of range')
            return seek_camera_data, MLX_data, senxor_m08_data, senxor_m16_data, realsense_depth_data, realsense_color_data
        try:
            seek_camera_frame = np.load(os.path.join(seek_thermal_folder, realsense_depth_files[slected_index]))
            seek_camera_data.append(seek_camera_frame)
        except:
            seek_camera_data.append(None)
        
        try:
            MLX_temperature_map = np.load(os.path.join(MLX_data_folder, realsense_depth_files[slected_index]))
            MLX_data.append(MLX_temperature_map)
        except:
            MLX_data.append(None)
            
        try:
            senxor_temperature_map_m08 = np.load(os.path.join(senxor_m08_folder, realsense_depth_files[slected_index]))
            senxor_m08_data.append(senxor_temperature_map_m08)
        except:
            senxor_m08_data.append(None)    
        
        try:
            senxor_temperature_map_m16 = np.load(os.path.join(senxor_m16_folder, realsense_depth_files[slected_index]))
            senxor_m16_data.append(senxor_temperature_map_m16)
        except:
            senxor_m16_data.append(None)
        try:
            realsense_depth = np.load(os.path.join(realsense_depth_folder, realsense_depth_files[slected_index]), allow_pickle=True)
            realsense_depth_data.append(realsense_depth)
        except:
            realsense_depth_data.append(None)
        try:
            realsense_color_image = np.load(os.path.join(realsense_color_folder, realsense_depth_files[slected_index]), allow_pickle=True)
            realsense_color_data.append(realsense_color_image)
        except:
            realsense_color_data.append(None)
        file_name_list.append(realsense_depth_files[slected_index])
    if file_name_list_flag:
        return seek_camera_data, MLX_data, senxor_m08_data, senxor_m16_data, realsense_depth_data, realsense_color_data, file_name_list
    return seek_camera_data, MLX_data, senxor_m08_data, senxor_m16_data, realsense_depth_data, realsense_color_data


def Get_COCO_pose_joint_label(joint_index):
    """
    Get the COCO pose joint label
    
    Parameters
    ----------
    joint_index : int
        The joint index
    
    Returns
    -------
    str
        The COCO pose joint label
    """
    COCO_pose_joint_labels = [
        "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
        "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
        "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
        "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
    ]
    return COCO_pose_joint_labels[joint_index]

COCO_Pose_Connections = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8),  # Arms
    (10, 11), (10, 12), (11, 13), (12, 14),  # Body
    (13, 15), (14, 16)  # Legs
]


def colorize_thermal_map(temperature_map, colormap='jet', vmin=None, vmax=None):
    """
    Colorize a thermal temperature map using a colormap.

    Parameters:
    temperature_map: 2D numpy array, the thermal temperature map.
    colormap: str, the name of the colormap to use (default: 'jet').
    vmin: float, minimum value for the colormap (default: min value in the map).
    vmax: float, maximum value for the colormap (default: max value in the map).

    Returns:
    colorized_map: 2D numpy array of shape (H, W, 4), representing the RGBA image.
    """
    # Normalize the temperature map to the range [0, 1]
    norm = plt.Normalize(vmin=vmin if vmin is not None else np.min(temperature_map),
                         vmax=vmax if vmax is not None else np.max(temperature_map))
    
    # Apply the colormap
    colormap = plt.get_cmap(colormap)
    colorized_map = colormap(norm(temperature_map))

    return colorized_map


def depth_map_to_point_cloud(depth_map, hfov_deg, vfov_deg):
        #testing
        print(f"hfov: {hfov_deg}, vfov: {vfov_deg};")
        import time
        time.sleep(30)
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


def calculate_iou(box0, box1):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    box0, box1: list or tuple of 4 elements (x1, y1, x2, y2)
                (x1, y1) is the top-left corner,
                (x2, y2) is the bottom-right corner.

    Returns:
    iou: float, the IoU value between the two boxes.
    """
    # Extract coordinates of the boxes
    x0_1, y0_1, x0_2, y0_2 = box0
    x1_1, y1_1, x1_2, y1_2 = box1

    # Calculate the coordinates of the intersection rectangle
    x_inter1 = max(x0_1, x1_1)
    y_inter1 = max(y0_1, y1_1)
    x_inter2 = min(x0_2, x1_2)
    y_inter2 = min(y0_2, y1_2)

    # Calculate the area of the intersection rectangle
    inter_width = max(0, x_inter2 - x_inter1)
    inter_height = max(0, y_inter2 - y_inter1)
    inter_area = inter_width * inter_height

    # Calculate the area of both bounding boxes
    box0_area = (x0_2 - x0_1) * (y0_2 - y0_1)
    box1_area = (x1_2 - x1_1) * (y1_2 - y1_1)

    # Calculate the area of the union
    union_area = box0_area + box1_area - inter_area

    # Calculate the IoU
    iou = inter_area / union_area if union_area != 0 else 0

    return iou