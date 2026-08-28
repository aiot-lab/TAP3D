import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import os
import argparse
from pathlib import Path
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import traceback

def strToNum(myNum):
    numfields = myNum.split("o")
    myNum = ".".join(numfields)
    return float(myNum)

class OutputVisualizer:
    def __init__(self):
        self.init_camera_figure()
        self.thermal_image_filedata={} #format: entry_name: [entry_dirpath (/home/shared/DeepTADAR/RawData1/U0_E0_1_sitting_1o3_none_2): [1.npy, 2.npy, ...]]
        self.depth_image_filedata = {}
        self.rgb_image_filedata = {}

    # for drawing on visualization
    def init_camera_figure(self):
        self.azim_temp=270
        self.vertices = np.array([
            [-100, 100, -100], [100, 100, -100], [100, 100, 60], [-100, 100, 60],  # Bottom face
            [0, -100, 0], [0, -100, 0], [0, -100, 0], [0, -100, 0]   # Top face
        ])
        self.faces = [
            [self.vertices[0], self.vertices[1], self.vertices[2], self.vertices[3]],  # Bottom
            [self.vertices[4], self.vertices[5], self.vertices[6], self.vertices[7]],  # Top
            [self.vertices[0], self.vertices[1], self.vertices[5], self.vertices[4]],  # Front
            [self.vertices[2], self.vertices[3], self.vertices[7], self.vertices[6]],  # Back
            [self.vertices[1], self.vertices[2], self.vertices[6], self.vertices[5]],  # Right
            [self.vertices[0], self.vertices[3], self.vertices[7], self.vertices[4]]   # Left
        ]

    # ==========================================================loading data============================================================
    #given a pickle path name, load predicts, labels and experiment config
    def load_data(self, log_path, target):
        with open(log_path, 'rb') as f:
            self.log = pickle.load(f)
        print("Finish loading the log file:"+log_path)
        # extract the fields we want to plot
        # target = 'U0_E0_1_lying_1o3_none_2'
        self.target = target
        self.predicts = self.log[target]["predicts"]
        self.labels = self.log[target]["labels"]
        self.camera_height = strToNum(self.target.split("_")[4])
        self.exp_config=self.log["exp_config"]
        return self.predicts, self.labels, self.exp_config
    
    def load_data_direct(self, log, target):
        self.predicts=log["predicts"]
        self.labels = log["labels"]
        self.exp_config = log["exp_config"]
        self.target = target

    # load depth and thermal image files 
    # task: load npy file names of corresponding folder that contains npy files of the target data entry (e.g. U0_E0_1...)
    # npy_dir_path is the directory where the npy files are contained
    # sensor_name is m08, m16, seek, etc.
    def load_ori_images(self, npy_dir_path, thermal_sensor_name, depth_sensor_name="realsense_depth", color_sensor_name="realsense_color"):
        npy_dir_name = npy_dir_path.split("/")[-1]
        self.thermal_image_filedata[npy_dir_name] = [os.path.join(npy_dir_path, thermal_sensor_name), os.listdir(os.path.join(npy_dir_path, thermal_sensor_name))]
        self.depth_image_filedata[npy_dir_name] = [os.path.join(npy_dir_path, depth_sensor_name), os.listdir(os.path.join(npy_dir_path, depth_sensor_name))]
        self.rgb_image_filedata[npy_dir_name] = [os.path.join(npy_dir_path, color_sensor_name), os.listdir(os.path.join(npy_dir_path, color_sensor_name))]
        
        self.thermal_image_filedata[npy_dir_name][1].sort()
        self.depth_image_filedata[npy_dir_name][1].sort()
        self.rgb_image_filedata[npy_dir_name][1].sort()

    # ==============================================preparing video output destination===============================================
    # init a video file to accept the converted frames
    def create_image_output_dest(self, video_path):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(video_path, fourcc, 20, (1500, 1000))

    # close video (output destination of converted frames)
    def close_image_output_dest(self):
        self.out.release()

    # ========================================================plotting========================================================================
    # given an axis, plot the 3d point cloud on the axis
    # point_cloud: points in one frame; ax: the axis for matplotlib; title: title for subplot; color: color of the point
    def plot_3d_point_cloud(self, point_cloud, max_num_persons, max_num_points, ax, title, threshold=0.5, color = "blue"):
        points_per_person = max_num_points + 1
        scatter_ret = None
        # Create a 3D plot
        #fig = plt.figure(figsize=(10, 8))
        #ax = fig.add_subplot(111, projection='3d')
        # Define colormap for different users
        colors = plt.cm.jet(np.linspace(0, 1, max_num_persons))
        # Plot points for each person
        for person_idx in range(max_num_persons):
            # Extract points for this person (assuming each person has max_num_points)
            start_idx = person_idx * points_per_person
            end_idx = start_idx + points_per_person
            
            indicator_idx = (person_idx + 1) * points_per_person - 1
            #print(f"point_cloud shape: {point_cloud.shape}")
            #print(f"point_cloud indicator point at point_cloud[0, {indicator_idx}]: {point_cloud[0, indicator_idx]}")
            indicator_point = point_cloud[0, indicator_idx]
            if indicator_point > threshold: 
                # Get points for this person
                person_points = point_cloud[ :, start_idx:end_idx]
                
                # Reshape to get individual 3D points
                x = person_points[0, :]
                y = person_points[1, :]
                z = person_points[2, :]
                
                # Filter out points where all coordinates are 0
                valid_points = ~((x < 5) & (y < 5) & (z < 5) & (x > -5) & (y > -5) & (z > -5))
                x_valid = x[valid_points]
                y_valid = y[valid_points]
                y_valid = -y_valid  
                z_valid = z[valid_points]
                
                if len(x_valid) > 0:  # Only plot if there are valid points
                    scatter_ret = ax.scatter(x_valid, z_valid, y_valid, #c=[colors[person_idx]], 
                            label=f'Person {person_idx+1}', alpha=0.5, s=1, c=color)
        # Add indicator points with a different marker and size
        # Add indicator points with their values in the right bottom of the plot
        for person_idx in range(max_num_persons):
            indicator_idx = (person_idx + 1) * points_per_person - 1
            indicator_value = point_cloud[0, indicator_idx]
            
            # Position text in the right bottom corner with some offset for each person
            x_pos = ax.get_xlim()[1] * 0.8  # Right side
            y_pos = ax.get_ylim()[0] * 0.9  # Bottom
            z_pos = ax.get_zlim()[0] + person_idx * (ax.get_zlim()[1] - ax.get_zlim()[0]) * 0.05
            
            # Add the indicator value text
            #ax.text(x_pos, y_pos, z_pos, f"Person {person_idx+1}: {indicator_value:.2f}", 
                    #color=colors[person_idx], fontsize=10, fontweight='bold')
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        # set the axis limit
        ax.set_xlim(-2000, 2000)
        ax.set_ylim(0, 4000)
        ax.set_zlim(-1000*self.camera_height, 1000)
        # Add legend
        # if max_num_persons > 0:
        #     ax.legend()
        #plt.tight_layout()
        #plt.show()    
        return scatter_ret

    def draw_overlap(self, axis, predict_depth_map, label_temp, exp_config, azim=-80, elev=30, roll=0):
        pred_scatter = self.plot_3d_point_cloud(predict_depth_map, exp_config['max_num_persons'], exp_config['max_num_points'] , axis, threshold=0.0, title="", color="blue")
        gt_scatter = self.plot_3d_point_cloud(label_temp, exp_config['max_num_persons'], exp_config['max_num_points'] , axis, threshold=0.0, title="prediction+ground truth", color="orange")
        if pred_scatter !=None and gt_scatter != None:
            axis.legend([pred_scatter, gt_scatter], ["Predictions", "Ground Truth"],loc='upper right')
    
    # draw predict, ground truth and overlay on three separate axes of a plt figure
    # arguments: axis: returned by plt.subplot(1, 3)
    # predict_depth_map: one frame of point clouds, predict
    # label_temp: one frame of point cloud, label
    # exp_config: the experiment config
    # azim, elev, roll: different angles for viewing the data
    def draw_pgo(self, axis, predict_depth_map, label_temp, exp_config, azim=-80, elev=30, roll=0):
        self.plot_3d_point_cloud(predict_depth_map, exp_config['max_num_persons'], exp_config['max_num_points'] , axis[0], threshold=0.0, title="predict", color="blue")
        # test on 2nd axis
        self.plot_3d_point_cloud(label_temp, exp_config['max_num_persons'], exp_config['max_num_points'] , axis[1], threshold=0.0, title="ground truth", color="orange")
        # both on third axis
        pred_scatter=self.plot_3d_point_cloud(predict_depth_map, exp_config['max_num_persons'], exp_config['max_num_points'] , axis[2], threshold=0.0, title="", color="blue")
        gt_scatter=self.plot_3d_point_cloud(label_temp, exp_config['max_num_persons'], exp_config['max_num_points'] , axis[2], threshold=0.0, title="both", color="orange")
        
        if pred_scatter !=None and gt_scatter != None:
            axis[2].legend([pred_scatter, gt_scatter], ["Predictions", "Ground Truth"],loc='upper right')

        for i in range(3):
            # set view
            axis[i].view_init(elev=elev, azim=azim, roll=roll)
            # add camera
            axis[i].add_collection3d(Poly3DCollection(
                verts=self.faces, 
                facecolors='gray', 
                linewidths=1, 
                edgecolors='black', 
                alpha=1  # Transparency (0=invisible, 1=opaque)
            ))
    
    # draw Temperature, rgb, depth on given axix
    def draw_trd(self, axis, thermal_path, rgb_path, depth_path):
        #print(thermal_path, depth_path)
        thermal_image = np.load(thermal_path)
        axis[0].imshow(thermal_image)
        axis[0].set_title("thermal")
        rgb_image = np.load(rgb_path)
        rbg_image = rgb_image[:, :, [2, 0, 1]]
        axis[1].imshow(rbg_image)
        #print(rgb_image.shape)
        axis[1].set_title("rgb")
        depth_image = np.load(depth_path)
        axis[2].imshow(depth_image)
        axis[2].set_title("depth")

    # given a path to a npy file, visualize it on the axis.
    def draw_npy(self, axis, npy_file_path):
        image = np.load(npy_file_path)
        axis.imshow(image)

    # =====================================encapsulated functions, to be called for external usages=====================================
    def loaded_data_to_video_overlap(self, rotatedeg=1, stride=1):
        azim_temp = self.azim_temp
        for batch in tqdm(range(len(self.predicts))):
            batch_size = len(self.predicts[batch])
            for frame in range(0, len(self.predicts[batch]), stride):
                # get the target point cloud (shape: 3x15005)
                predict_depth_map = np.array(self.predicts[batch][frame])
                label_temp = np.array(self.labels[batch][frame])
                exp_config = self.exp_config
                # plot axes
                fig, axs = plt.subplots(1, 1, figsize=(15, 10), subplot_kw={'projection': '3d'})
                self.draw_overlap(axs, predict_depth_map, label_temp, exp_config, azim=azim_temp)
                azim_temp += rotatedeg
                azim_temp %= 360
                # draw on canvas
                fig.suptitle(f"data entry: {self.target}; frame: {batch*batch_size+frame}", y=1.0)
                fig.canvas.draw()
                img_plot = np.array(fig.canvas.renderer.buffer_rgba())
                self.out.write(cv2.cvtColor(img_plot, cv2.COLOR_RGB2BGR))
                plt.close(fig)
                
    # to video, containing three subplots: P (predict) G (ground truth) O (overlay predict and ground truth)
    # this requires self.out already contains the loaded data (predicts, labels, exp_config)
    # use: self.load_data() to load the required data (if data has not been loaded yet)
    def loaded_data_to_video_pgo(self, rotatedeg=1, stride=1):
        # output the the video file out
        # for i in tqdm(range(len(predicts))):
        #for batch in tqdm(range(len(predicts))):
        azim_temp = self.azim_temp
        for batch in tqdm(range(len(self.predicts))):
            batch_size = len(self.predicts[batch])
            for frame in range(0, len(self.predicts[batch]), stride):
                # get the target point cloud (shape: 3x15005)
                predict_depth_map = np.array(self.predicts[batch][frame])
                label_temp = np.array(self.labels[batch][frame])
                exp_config = self.exp_config
                # plot axes
                fig, axs = plt.subplots(1, 3, figsize=(15, 10), subplot_kw={'projection': '3d'})
                self.draw_pgo(axs, predict_depth_map, label_temp, exp_config, azim=azim_temp)
                azim_temp += rotatedeg
                azim_temp %= 360
                # draw on canvas
                fig.suptitle(f"data entry: {self.target}; frame: {batch*batch_size+frame}", y=0.80)
                fig.canvas.draw()
                img_plot = np.array(fig.canvas.renderer.buffer_rgba())
                self.out.write(cv2.cvtColor(img_plot, cv2.COLOR_RGB2BGR))
                plt.close(fig)
        
    # to video, containing five subplots: P (predict) G (ground truth) O (overlay predict and ground truth) T (thermal array) R (RGb) D (depth image)
    # make sure already used load_data() and load_ori_images() to prepare the required data
    # also, make sure we already initialized and opened self.out, which is the output video file
    def loaded_data_to_video_pgotrd(self, data_entry_name, rotatedeg=1, stride=1):
        azim_temp = self.azim_temp
        thermal_image_namelist = self.thermal_image_filedata[data_entry_name][1]
        thermal_image_basedir = self.thermal_image_filedata[data_entry_name][0]
        depth_map_namelist = self.depth_image_filedata[data_entry_name][1]
        depth_map_basedir = self.depth_image_filedata[data_entry_name][0]
        rgb_image_namelist = self.rgb_image_filedata[data_entry_name][1]
        rgb_image_basedir = self.rgb_image_filedata[data_entry_name][0]

        for batch in tqdm(range(len(self.predicts))):
        #for batch in tqdm(range(1)):
            batch_size = len(self.predicts[batch])
            for frame in range(0, len(self.predicts[batch]), stride):
            #for frame in range(1):

                # get the target point cloud (shape: 3x15005)
                predict_depth_map = np.array(self.predicts[batch][frame])
                label_temp = np.array(self.labels[batch][frame])
                exp_config = self.exp_config

                # get the original images paths
                current_frame_index = batch*batch_size+frame
                current_thermal_filepath = thermal_image_basedir + "/" + thermal_image_namelist[current_frame_index]
                current_rgb_filepath = rgb_image_basedir + "/" + rgb_image_namelist[current_frame_index]
                current_depth_filepath = depth_map_basedir + "/" + depth_map_namelist[current_frame_index]
                # print("========================thermal name: ", current_thermal_filepath)
                # print("========================depth name: ", current_depth_filepath)
                

                # plot axes
                fig = plt.figure(figsize=(15, 10))
                gs = fig.add_gridspec(2, 3)
                ax0 = fig.add_subplot(gs[0, 0])
                ax1 = fig.add_subplot(gs[0, 1])
                ax2 = fig.add_subplot(gs[0, 2])

                ax3 = fig.add_subplot(gs[1, 0], projection='3d')
                ax4 = fig.add_subplot(gs[1, 1], projection='3d')
                ax5 = fig.add_subplot(gs[1, 2], projection='3d')

                
                axs_row1 = [ax0, ax1, ax2]
                axs_row2 = [ax3, ax4, ax5]

                # plot 3d point cloud
                self.draw_pgo(axs_row2, predict_depth_map, label_temp, exp_config, azim=azim_temp)
                self.draw_trd(axs_row1, current_thermal_filepath, current_rgb_filepath, current_depth_filepath)

                azim_temp += rotatedeg
                azim_temp %= 360

                # draw on canvas
                fig.suptitle(f"data entry: {data_entry_name}; frame: {batch*batch_size+frame}", y=0.95)
                fig.canvas.draw()
                img_plot = np.array(fig.canvas.renderer.buffer_rgba())
                self.out.write(cv2.cvtColor(img_plot, cv2.COLOR_RGB2BGR))
                plt.close(fig)

    # log_path: path to the pkl file; 
    # target: the target data entry name
    # rotatedeg: degree of rotation per frame
    # P (predict) G (ground truth) O (overlay predict and ground truth)
    def convert_single_file_pgo(self, log_path, target, rotatedeg=0, stride=1):
        #load the file
        predicts, labels, exp_config = self.load_data(log_path, target)
        # save as a avi video
        video_path = log_path.replace('.pkl', '-short.avi')
        # create image output destinatio video file, write visualized images to file, close video file
        self.create_image_output_dest(video_path)
        self.loaded_data_to_video_pgo(rotatedeg=rotatedeg, stride=stride)
        self.close_image_output_dest()

    # given path to (path to pkl log file for the data entry, 
    # path to the data entry's original data, target thermal array name, 
    # rotation degree of the 3d images per frame)
    def convert_single_file_pgotd(self, log_path, ori_data_path, target, rotatedeg=0, stride=1):
        #load the file
        predicts, labels, exp_config = self.load_data(log_path, target)
        self.load_ori_images(ori_data_path, "senxor_m16")

        # save as a avi video
        video_path = log_path.replace('.pkl', '-short.avi')

        # create image output destinatio video file, write visualized images to file, close video file
        self.create_image_output_dest(video_path)
        self.loaded_data_to_video_pgotrd(target, rotatedeg=rotatedeg, stride=stride)
        self.close_image_output_dest()
    
    

if __name__=="__main__":
    #usage: python output_check_visualize.py --pkl_source_file logs/m08/unet_m08_unet_like_0618231718/ --save_file log_video.avi --sensor_type senxor_m08 --stride 10
    #usage: python output_check_visualize.py --pkl_source_file logs/seekthermal/unet_seekthermal_init_unet_like_0618174207/ --save_file log_video.avi --sensor_type seek_thermal --stride 1
    #usage: python output_check_visualize.py --pkl_source_file logs/m16/unet_m16_unet_like_0618231747/ --save_file log_video.avi --sensor_type senxor_m16 --stride 10
    #usage: python output_check_visualize.py --pkl_source_file logs/m08/resnet18_m08_init_resnet18_0625173503/ --save_file log_video.avi --sensor_type senxor_m08 --stride 10

    #obtain arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_source_file", type=str, help="path to the file where video is saved", default = "")
    parser.add_argument("--save_file", type=str, help="video file name", default = "log_video.avi")
    parser.add_argument("--visualize_specified", type=int, help="0 for processing all entries in specified folder, 1 for processing a specific entry", default = 0)
    parser.add_argument("--ori_data_dirbase", type=str, help="path to original data base dir, such as: RawData1/", default = "RawData1/")
    parser.add_argument("--sensor_type", type=str, required=True)
    parser.add_argument("--stride", type=int, help="stride of the video, 1 for every frame, 2 for every other frame, etc.", default = 1)
    args = parser.parse_args()
    base_dir = args.pkl_source_file
    video_file_name = args.save_file
    stride = args.stride

    rawdata_dirs = [os.path.abspath(i) for i in os.listdir(".") if "RawData" in i]
    rawdata_dirs.sort()
    rawdata_contents = []
    for rawdatadir in rawdata_dirs:
        contents = os.listdir(rawdatadir)
        rawdata_contents.append(contents)
    
    try:
        if args.visualize_specified==0:
            if Path(base_dir+video_file_name).exists():
                print("already having the video file: "+base_dir+video_file_name)
                #exit()
            ov = OutputVisualizer()
            # initialize the avi file to save data
            video_path = base_dir+video_file_name
            ov.create_image_output_dest(video_path)
            # list all contents in the folder that contains all data entries
            load_paths=os.listdir(base_dir)
            load_paths.sort()
            # loop through every folder that contains all data entries
            for i in range(len(load_paths)):
                # parse the log_path and target
                name = load_paths[i]
                if name.split(".")[-1]!="pkl":
                    continue
                log_path=args.pkl_source_file+"/"+name
                # data_entry_name = "-".join(name.split("-")[1:]).split(".")[0] #U0_E0_1
                data_entry_name = name.split(".")[0]
                # target data entry name
                sensor = args.sensor_type # senxor_m16
                enclosing_rawdata_path = "" # RawData0? 1? 2? ...
                for dirpathname, dircontentlist in zip(rawdata_dirs, rawdata_contents):
                    if data_entry_name in dircontentlist:
                        enclosing_rawdata_path = dirpathname
                        break
                if enclosing_rawdata_path=="":
                    print(f"cannot find the path to original data of entry: {data_entry_name}")
                    continue
                ori_data_path = os.path.join(enclosing_rawdata_path, data_entry_name) #/home/shared/DeepTADAR/RawData1 + U0_E0_1_standing_1o3_none_2
                print("path of the entry: ", ori_data_path)
                ov.load_ori_images(ori_data_path, args.sensor_type)
                ov.load_data(log_path, data_entry_name)
                print("target:", data_entry_name)
                ov.loaded_data_to_video_pgotrd(data_entry_name, stride=stride)           
            ov.close_image_output_dest()

        #convert a single pkl file to video
        if args.visualize_specified==1:
            log_path = base_dir #"logs/m16/unet_like_2025032500000/unet_m16_unet_like_0325214810_test-U0_E0_1_standing_1o3_none_2.pkl"
            ori_data_path = args.ori_data_dirbase #"RawData1/U0_E0_1_standing_1o3_none_2"
            target = ori_data_path.split("/")[-1]
            ov = OutputVisualizer()
            ov.convert_single_file_pgotd(log_path, ori_data_path, target, rotatedeg=1, stride=stride)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        # exit(1)

        
        


