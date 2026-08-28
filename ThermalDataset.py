import os
import numpy as np
import pandas as pd
import pickle   
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import yaml
from torch.utils.data import Dataset, DataLoader


def virtual_dataloader(data_config, sensor_name, chunk_size=10, tqdm_disable=False):
    '''
    Load the data sample paths not the data itself
    
    Args:
    data_config: dictionary, the configuration of the dataset, i.e., selected subfolders, dataset path, etc.
    sensor_name: str, the name of the sensor, e.g., 'seek_thermal', 'senxor_m08', 'senxor_m16'  
    chunk_size: int, the number of samples in each training sample. Default: 10 (since our data collecting system collects about 10 samples per second)
    tqdm_disable: bool, whether to disable the tqdm progress bar
    '''
    data_path = data_config['dataset_path'] # the folder containing all the data
    sub_folder_list = data_config['sub_folder_list'] # the name of the subfolders containing the data, each subfolder contains one recording session
    if len(sub_folder_list) == 0:  # load all the recordings in the data_path
        sub_folder_list = os.listdir(data_path)

    data_paths = []
    label_paths = []
    attributes = []  # we can have the experiment information based on the subfolder name, like the envoriment id, the user id, etc.
    for sub_folder in tqdm(sub_folder_list, disable=tqdm_disable):
        sub_folder_path = os.path.join(data_path, sub_folder)
        sensor_data_path = os.path.join(sub_folder_path, sensor_name)  # each subfolder contains the data of several sensors
        if os.path.isdir(sensor_data_path):
            file_list = os.listdir(sensor_data_path)
            data_list = [f for f in file_list if f.endswith('.npy')]   # the data files are saved in .npy format, the corresponding label files are saved in .pkl format
            data_list.sort()
            label_list = [f.replace('.npy', '.pkl') for f in data_list]
            num_samples = len(data_list)    # the number of samples in the recording session
            # seperate the data into chunks
            for i in range(0, num_samples, chunk_size):
                if i + chunk_size < num_samples:
                    data_paths.append([os.path.join(sensor_data_path, f) for f in data_list[i:i+chunk_size]])
                    label_paths.append([os.path.join(sensor_data_path, f) for f in label_list[i:i+chunk_size]])
                    attributes.append(sub_folder)
                else:
                    break
    return data_paths, label_paths, attributes
            


class MyData(Dataset):
    def __init__(self,exp_config, data_config, is_training=True):
        self.exp_config = exp_config
        self.data_config = data_config
        
        self.sensor_name = exp_config['sensor_name'] 
        self.max_num_points = exp_config['max_num_points']   # the maximum number of points in the point cloud, used to sampling/pad the point cloud
        self.max_num_persons = exp_config['max_num_persons']  # the maximum number of persons in the environment
        self.chunk_size = exp_config['chunk_size']
        tqdm_disable = exp_config['tqdm_disable']
        self.preload = exp_config['preload'] # whether to load all the data into memory
        self.label_maker = label_maker(exp_config)
        self.normalize_data = exp_config.get('normalize_data', False) # we do not change the temperature by default
        if self.normalize_data:
            print('Data is normalized')
        
        data_paths, label_paths, attributes = virtual_dataloader(data_config, self.sensor_name, self.chunk_size, tqdm_disable)
        
        if self.preload:
            print('Loading data into memory...')
            self.data = []
            self.labels = []
            self.attributes = []
            for i in tqdm(range(len(data_paths)), disable=tqdm_disable):
                data = [np.load(f) for f in data_paths[i]]
                label = [pickle.load(open(f, 'rb')) for f in label_paths[i]]
                self.data.append(data)
                self.labels.append(label)
                self.attributes.append(attributes[i])
        else:
            self.data_paths = data_paths
            self.label_paths = label_paths
            self.attributes = attributes
            print('Data paths are loaded')
            print('Number of samples: ', len(self.data_paths))

    def __len__(self):
        if self.preload:
            return len(self.data)
        else:
            return len(self.data_paths)
    
    def __getitem__(self, idx):
        if self.preload:
            data_sample, label_sample, attribute = self.data[idx], self.labels[idx], self.attributes[idx]
        else:
            data_sample, label_sample, attribute = self._load_data(idx)          
        data_sample = np.stack(data_sample, axis=0)  # shape: [num_samples, *data_shape]
        # use the last element in the label list as the final label
        label_sample = label_sample[-1]
        num_persons = label_sample['num_persons']
        depth_person = label_sample['depth_person']
        depth_mask_person = label_sample['depth_mask_person']
        point_cloud_person = label_sample['point_cloud_person']
        pose_2D_person = label_sample['2D_pose_person']  
        label = self.label_maker.forward(num_persons, depth_person, depth_mask_person, point_cloud_person, pose_2D_person)
        user_id, env_id, num_people, Activity_str, device_height = self._attribute_parse(attribute)  
        activity_id = self._activity2label(Activity_str)
        
        label = torch.tensor(label, dtype=torch.float32)
        data_sample = torch.tensor(data_sample, dtype=torch.float32)
        if self.normalize_data:
            max_temp = data_sample.max()
            min_temp = data_sample.min()
            data_sample = (data_sample - min_temp) / (max_temp - min_temp)
        return data_sample, label, user_id, env_id, num_people, activity_id, device_height
    
    def _load_data(self, idx):
        # load the data from the self.data_paths and self.label_paths
        data = [np.load(f) for f in self.data_paths[idx]]
        label = [pickle.load(open(f, 'rb')) for f in self.label_paths[idx]]
        return data, label, self.attributes[idx]
    
    def _attribute_parse(self, attribute):
        # parse the attribute to get the experiment information, the details is in collection_recording.xlsx
        # example: U0o1o2_E2_3_walking_1o3_none_0
        # speerate the attribut string by '_', and get the information
        infos = attribute.split('_')
        user_str = infos[0][1:]  # the user id. if multiple users will have 'o' seperated, then we keep the first one
        if 'x' in user_str:
            user_id = np.nan  # unknown user 
        else:
            if 'o' in user_str:
                user_str = user_str.split('o')[0]
            try:
                user_id = int(user_str)
            except:
                user_id = np.nan  # there is no user in the recording
                
        env_id = int(infos[1][1:])  # the environment id
        if 'x' in infos[2]:
            num_people = np.nan  # unknown number of people 
        else:
            num_people = int(infos[2])  # the number of people in the environment
            
        Activity_str = infos[3]  # the activity of the people. if multiple activities will have 'o' seperated, then we keep the first one
        if Activity_str == 'none':
            Activity_str = ""
        else:
            if 'cooking' in Activity_str:
                Activity_str = 'cooking'
            else:
                if 'o' in Activity_str:
                    Activity_str = Activity_str.split('o')[0]
                
        device_height_str = infos[4]  # the height of the device. We use 'o' to represent '.' the unit is meter
        if device_height_str == 'handheld':
            device_height = 0.0
        else:
            if 'o' in device_height_str:
                device_height_str = device_height_str.replace('o', '.')
            device_height = float(device_height_str)
        return user_id, env_id, num_people, Activity_str, device_height

        
    def _activity2label(self, activity_str):
        # convert the activity string to the label
        activity_dict = {'walking': 0, 'standing': 1, 'sitting': 2, 'lying': 3, 'cooking': 4, '' : np.nan}
        return activity_dict[activity_str]
    
    
# label creator: making the label for different tasks/models
class label_maker():
    def __init__(self,exp_config):
        self.max_num_persons = exp_config['max_num_persons']
        self.label_type = exp_config['label_type']   # 'depth_mask_3C'
        self.max_num_points = exp_config['max_num_points']
        if exp_config['sensor_name'] == 'seek_thermal':
            self.width = 200
            self.height = 150
        elif exp_config['sensor_name'] == 'senxor_m08':
            self.width = 80
            self.height = 62
        elif exp_config['sensor_name'] == 'senxor_m16':
            self.width = 160
            self.height = 120
        else:
            raise ValueError('Invalid sensor name')
        
        if self.label_type == 'depth_mask_3C':
            self.convertor = self.depth_mask_3C  # the function to convert the label to the desired format
        elif self.label_type == 'point_cloud_3C':
            self.convertor = self.point_cloud_3C
        else:
            raise ValueError('Invalid label type')
    
    def depth_mask_3C(self,depth_person, depth_mask_person):
        '''
        Create the 3-channel depth mask for the person/persons: 
            the first channel is the depth map containing the person(s), where the depth value is the distance from the camera to the person, and the background is 0. Note that if the pixel has more than one person, we use the depth value of the nearest person.
            the second channel is the indicator map, where the region of the first person is 1, the region of the second person is 2, etc.
            the third channel is the binary mask of foreground (person) and background, where the person region is 1, the background region is 0
        Args:
        depth_person: list of the average depth of the person(s)
        depth_mask_person: list of np.array, the depth mask of the person(s), each element is the depth mask of one person
        Returns:
        depth_mask_3C: np.array, the 3-channel depth mask of the person(s)
        '''
        # Initialize the output 3-channel mask
        depth_map = np.zeros((self.height, self.width), dtype=np.float32)  # Channel 1: Depth map
        indicator_map = np.zeros((self.height, self.width), dtype=np.int32)  # Channel 2: Indicator map
        binary_mask = np.zeros((self.height, self.width), dtype=np.int32)  # Channel 3: Binary mask
        
        if  len(depth_mask_person) == 0:
            # If no person is detected, return an empty depth mask
            depth_mask_3C = np.zeros((3, self.height, self.width), dtype=np.float32)
        else:
            # start with the person with the nearest depth: get the order of the person based on the average depth
            order = np.argsort(depth_person)
            order = order[:self.max_num_persons]  # only keep the first max_num_persons persons
            for order_i, i in enumerate(order):
                person_mask = depth_mask_person[i]
                person_mask_valid = person_mask > 0
                # Update the depth map and indicator map only if the current person is nearer
                update_mask = ((depth_map == 0) | ((person_mask < depth_map) & (person_mask > 0))) & (person_mask >0)    # 20 mm depth threshold
                # update_mask = ((person_mask < depth_map) & (person_mask > 0))   # 20 mm depth threshold
                depth_map = np.where(update_mask, person_mask, depth_map)
                indicator_map = np.where(update_mask, order_i + 1, indicator_map)
                # Update the binary mask (foreground is 1, background is 0)
                binary_mask = np.where(person_mask_valid, 1, binary_mask)
            # Stack the three channels to create the 3-channel depth mask
            depth_mask_3C = np.stack([depth_map, indicator_map, binary_mask], axis=0)
        return depth_mask_3C
    
    def point_cloud_3C(self,depth_person, point_cloud_person):
        '''
        Create a 3-channel point cloud representation for the person/persons:
            Each point cloud is padded to max_num_points and includes an indicator point
            The point clouds are concatenated in order of increasing depth (nearest person first)
            
        Args:
        depth_person: list of the average depth of the person(s)
        point_cloud_person: list of np.array, the point cloud of the person(s), each element is the point cloud of one person
        
        Returns:
        point_cloud_3C: np.array, the 3-channel point cloud representation of shape (3, (max_num_points + 1) * max_num_persons)
        '''
        # Initialize the output array with zeros
        output_shape = (3, (self.max_num_points + 1) * self.max_num_persons)
        point_cloud_3C = np.zeros(output_shape, dtype=np.float32)
        
        if len(point_cloud_person) == 0:
            # If no person is detected, return an empty point cloud
            return point_cloud_3C
        else:
            # Sort persons by depth (nearest first)
            order = np.argsort(depth_person)
            if len(order) > self.max_num_persons:
                order = order[:self.max_num_persons]  # only keep the first max_num_persons persons
            
            current_idx = 0
            for i in order:
                # Get the point cloud for this person
                person_points = point_cloud_person[i]
                person_points = self._point_cloud_sampling(person_points)
                # Copy the points to the output array
                point_cloud_3C[:, current_idx:current_idx + self.max_num_points] = person_points
                # Add the indicator point (1,0,0) to show this is a person
                indicator_idx = current_idx + self.max_num_points
                point_cloud_3C[0, indicator_idx] = 1.0  # Set x=1 to indicate a person
                # Move to the next person's section in the output array
                current_idx += (self.max_num_points + 1)
            return point_cloud_3C
            
    def forward(self, num_persons, depth_person, depth_mask_person, point_cloud_person, pose_2D_person):
        if self.label_type == 'depth_mask_3C':
            return self.convertor(depth_person, depth_mask_person)
        elif self.label_type == 'point_cloud_3C':
            return self.convertor(depth_person, point_cloud_person)
        else:
            raise ValueError('Invalid label type')
    
    def _point_cloud_sampling(self, point_cloud):
        num_points = point_cloud.shape[0]
        # if the number of points is less than the max_num_points, we pad the point cloud;
        # if the number of points is more than the max_num_points, we sample the point cloud
        if num_points == 0:
            return np.zeros((3, self.max_num_points))
        if num_points < self.max_num_points:
            pad_num = self.max_num_points - num_points
            pad = np.zeros((pad_num, 3))
            point_cloud = np.concatenate([point_cloud, pad], axis=0)  # shape: [max_num_points, 3]
        else:
            idx = np.random.choice(num_points, self.max_num_points, replace=False)
            point_cloud = point_cloud[idx]    
        return point_cloud.T  # Return the sampled/padded point cloud with shape [3, max_num_points]


if __name__ == "__main__":
    # the exp_confi file at exp_configs/sample_exp_config.yaml
    exp_config = yaml.load(open('exp_configs/sample_exp_config.yaml'), Loader=yaml.FullLoader)  
    # the data_config file at data_configs/sample_data_config.yaml
    data_config = yaml.load(open('data_configs/sample_data_config.yaml'), Loader=yaml.FullLoader)

    # create a dataset object
    dataset = MyData(exp_config, data_config)
    # get the data from the dataset object, this is a pytorch dataset objectpr
    print(len(dataset))
    data = dataset[1000]

    data_sample, label, user_id, env_id, num_people, activity_id, device_height = data
    print(data_sample.shape, label.shape, user_id, env_id, num_people, activity_id, device_height)