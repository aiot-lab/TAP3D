import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from Metrics import PointCloudMetrics
import torch
import argparse
from tqdm import tqdm


def calculate_tp_fp_fn(pred_indicator, target_indicator):
    pred_detected = (pred_indicator > 0.5).float()
    gt_detected = (target_indicator > 0.5).float()
    true_positives = (pred_detected * gt_detected).sum()
    false_positives = (pred_detected * (1 - gt_detected)).sum()
    false_negatives = ((1 - pred_detected) * gt_detected).sum()
    return true_positives, false_positives, false_negatives



if __name__ == "__main__":
    # generate the metrics recording of the model and saved in a pickle file
    # the metrics recording is a dictionary with the following keys:
    # 'true_positives', 'false_positives', 'false_negatives', 'chamfer_distance', 'user_id', 'env_id', 'num_people', 'activity_id', 'device_height', 'x_average', 'y_average', 'z_average'
    
    # python output_metric_calc.py --log_folder_path logs/m08/unet_m08_pilot_unet_like_0626101104
    # python output_metric_calc.py --log_folder_path logs/m08/model3_m08_thermo_pt_0819203728
    # python output_metric_calc.py --log_folder_path logs/m08/model2_m08_thermo_pt_0819203650
    # python output_metric_calc.py --log_folder_path logs/m08/model1_m08_thermo_pt_0819203603
    # python output_metric_calc.py --log_folder_path logs/m08/model0_m08_thermo_pt_0819092207
    # python output_metric_calc.py --log_folder_path logs/m16/model3_m16_thermo_pt_0820225212
    # python output_metric_calc.py --log_folder_path logs/m16/model3_m16_thermo_pt_0820152400
    # python output_metric_calc.py --log_folder_path logs/seekthermal/model3_seekthermal_thermo_pt_0820225255
    # python output_metric_calc.py --log_folder_path logs/m08/TAP3D_m08_thermo_pt_0823085239
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_folder_path', type=str, default=None, required=True)
    args = parser.parse_args()
    
    # load the existed metrics
    final_metric_save_path = os.path.join(args.log_folder_path, 'final_metric.pickle')
    processed_file_name = []
    existed_metrics = None
    if os.path.exists(final_metric_save_path):
        print(f"Final metric file already exists at {final_metric_save_path}")
        existed_metrics = pickle.load(open(final_metric_save_path, 'rb'))
        processed_file_name = existed_metrics['file_name']

    Log_folder_path = args.log_folder_path
    pickle_files = [f for f in os.listdir(Log_folder_path) if f.endswith('.pkl')]
    # only keep the files that not in the existed_metric
    pickle_files = [f for f in pickle_files if f not in processed_file_name]
    num_files = len(pickle_files)
    
    print('the number of files already processed: ', len(processed_file_name))
    print('the number of files to be processed: ', num_files)

    if existed_metrics is None:
        existed_metrics = {
            'file_name': [],
            'metrics': [],
            'no_person_frame_count': [],
        }
    
    for file_index in tqdm(range(num_files)):
        file_path = os.path.join(Log_folder_path, pickle_files[file_index])
        file_name = pickle_files[file_index]
        
        true_positive_list = []
        false_positive_list = []
        false_negative_list = []
        chamfer_distance_list = []
        hausdorff_distance_list = []
        point2point_distance_list = []
        user_id_list = []
        env_id_list = []
        num_people_list = []
        activity_id_list = []
        device_height_list = []
        x_average_list = []
        y_average_list = []
        z_average_list = []
        
        no_person_count = 0
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        recording_name = pickle_files[file_index].split('.')[0]
        recording = data[recording_name]
        exp_config = data['exp_config']
        metric_method = PointCloudMetrics(exp_config)
        
        predicts = recording['predicts']
        num_batch = len(predicts)
        attributes = recording['attributes']
        labels = recording['labels']
        metrics = recording['metrics']

        # process each batch
        for batch_idx in range(num_batch):
            predict_batch = predicts[batch_idx]
            batch_size = len(predict_batch)
            attributes_batch = attributes[batch_idx]
            user_id_batch, env_id_batch, num_people_batch, activity_id_batch, device_height_batch = attributes_batch
            labels_batch = labels[batch_idx]

            predict_batch = torch.tensor(predict_batch)
            labels_batch = torch.tensor(labels_batch)

            # calculate the metrics for each sample
            for i in range(batch_size):
                predict_batch_i = predict_batch[i,:,:].unsqueeze(0)   # shape: [1, 3, max_num_points]
                labels_batch_i = labels_batch[i,:,:].unsqueeze(0)   # shape: [1, 3, max_num_points]
                user_id_batch_i = user_id_batch[i]
                env_id_batch_i = env_id_batch[i]
                num_people_batch_i = num_people_batch[i]
                activity_id_batch_i = activity_id_batch[i]
                device_height_batch_i = device_height_batch[i]
                target_indicator, target_cloud_point, pred_indicator, pred_cloud_point, present_persons = metric_method._data_parsing(predict_batch_i, labels_batch_i)
                '''
                target_indicator: shape: [1, max_num_persons]
                target_cloud_point: shape: [1, 3, max_num_points, max_num_persons]
                pred_indicator: shape: [1, max_num_persons]
                pred_cloud_point: shape: [1, 3, max_num_points, max_num_persons]
                present_persons: list of indices of the present persons
                '''
                if len(present_persons) > 0:    
                    # find the position of the fist person
                    target_cloud_point_present = target_cloud_point[:,:,:,present_persons[0]]
                    first_person_points = target_cloud_point_present[0,:,:] # shape: [3, max_num_points]
                    first_person_points_mask = torch.any(first_person_points != 0, dim=0)
                    first_person_points = first_person_points[:,first_person_points_mask]
                    x_average = first_person_points[0].mean()
                    y_average = first_person_points[1].mean()
                    z_average = first_person_points[2].mean()
                    
                    # Calculate individual metrics
                    true_positives, false_positives, false_negatives = calculate_tp_fp_fn(pred_indicator, target_indicator)
                    chamfer_dist = metric_method._chamfer_distance2(pred_cloud_point, target_cloud_point, target_indicator)
                    hausdorff_dist = metric_method.calculate_hausdorff_distance(pred_cloud_point, target_cloud_point, target_indicator)
                    point2point_dist = metric_method.calculate_point2point_distance(pred_cloud_point, target_cloud_point, target_indicator)
                    
                    true_positive_list.append(true_positives.item())
                    false_positive_list.append(false_positives.item())
                    false_negative_list.append(false_negatives.item())
                    chamfer_distance_list.append(chamfer_dist.item())
                    hausdorff_distance_list.append(hausdorff_dist.item())
                    point2point_distance_list.append(point2point_dist.item())
                    user_id_list.append(user_id_batch_i.item())
                    env_id_list.append(env_id_batch_i.item())
                    num_people_list.append(num_people_batch_i.item())
                    activity_id_list.append(activity_id_batch_i.item())
                    device_height_list.append(device_height_batch_i.item())
                    x_average_list.append(x_average.item())
                    y_average_list.append(y_average.item())
                    z_average_list.append(z_average.item())
                else:
                    # print(i, 'no person detected')
                    no_person_count += 1

        metrics = {
            'true_positives': true_positive_list,
            'false_positives': false_positive_list,
            'false_negatives': false_negative_list,
            'chamfer_distance': chamfer_distance_list,
            'hausdorff_distance': hausdorff_distance_list,
            'point2point_distance': point2point_distance_list,
            'user_id': user_id_list,
            'env_id': env_id_list,
            'num_people': num_people_list,
            'activity_id': activity_id_list,
            'device_height': device_height_list,
            'x_average': x_average_list,
            'y_average': y_average_list,
            'z_average': z_average_list,
        }
        existed_metrics['metrics'].append(metrics.copy())
        existed_metrics['file_name'].append(file_name)
        existed_metrics['no_person_frame_count'].append(no_person_count)
    pickle.dump(existed_metrics, open(final_metric_save_path, 'wb'))
