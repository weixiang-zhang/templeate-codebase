import os
import cv2 as cv
import torch
import imageio.v2 as imageio
import math
import copy
import random,string
import json
import numpy as np
# from PIL import Image, ImageDraw


input_img_path = "/home/ubuntu/projects/coding4paper/data/div2k/test_data/03.png"
output_path = os.path.join("/home/ubuntu/projects/coding4paper/projects/libinr/log/000_visualized", "visual_sample_points")

# random
# indicies_path = "/home/ubuntu/projects/coding4paper/projects/libinr/log/022_visualize_0.5_single_03/random_2024-10-08-16:17:34/record/03_indices.json"

# fm
indicies_path = "/home/ubuntu/projects/coding4paper/projects/libinr/log/022_visualize_0.5_single_03/fm_cur_2024-10-08-16:17:34/record/03_indices.json"

# es
# indicies_path = "/home/ubuntu/projects/coding4paper/projects/libinr/log/022_visualize_0.5_single_03/expansive_2024-10-08-16:17:34/record/03_indices.json"

# egra
# indicies_path = "/home/ubuntu/projects/coding4paper/projects/libinr/log/022_visualize_0.5_single_03/egra_2024-10-08-16:17:34/record/03_indices.json"

# soft
# indicies_path = "/home/ubuntu/projects/coding4paper/projects/libinr/log/022_visualize_0.5_single_03/soft_2024-10-08-16:17:34/record/03_indices.json"

# step-fm
# indicies_path = "/home/ubuntu/projects/coding4paper/projects/libinr/log/022_visualize_0.5_step_single_03/fm_cur_2024-10-08-17:04:48/record/03_indices.json"

# step-nmt
# indicies_path = "/home/ubuntu/projects/coding4paper/projects/libinr/log/022_visualize_0.5_step_single_03/nmt_incre_2024-10-08-17:57:15/record/03_indices.json"

os.makedirs(output_path, exist_ok=True)


purple=(112,77,168)
green1=(115,192,136)
white=(255,255,255)
black = (0,0,0)
# green2=(51,63,41)
color = purple

def gen_cur_time():
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

def gen_random_str(nums=4):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(nums))


def read_indices_from_json(json_path, epoch=500):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data[f"epoch_{epoch}"]

def random_sample(input_img_path, output_path, epoch=500,mark=''):
    bright = 0.3

    cur_time = gen_cur_time()
    img_id = os.path.basename(indicies_path).split("_")[0].strip() 
    strategy = os.path.basename(os.path.dirname(os.path.dirname(indicies_path))).split("_")[0].strip() 

    img_io = imageio.imread(input_img_path)
    # img_io = img_io[::2,::2]

    img = torch.from_numpy(img_io)
    h,w,c, = img.shape

    # torch random sample
    # _ratio = 0.3
    # _ratio_len  = int(h*w*_ratio)
    # _shuffled_index_arr = torch.randperm(h*w)
    # _indices = _shuffled_index_arr[:_ratio_len]
    # remian_indices = _shuffled_index_arr[_ratio_len:]

    
    indices = read_indices_from_json(indicies_path, str(epoch))
    all_nums = np.arange(h*w)
    remian_indices = np.setdiff1d(all_nums, indices)
    
    if strategy == "fm": # inverse
        mark_indices = remian_indices
    else:
        mark_indices = indices
    
    for i in range(w):
        for j in range(h):
            for r in range(3):
                img_io[i,j,r] = 0
        
    
    for index in mark_indices:
        i = math.floor(index / w)
        j = index % w
        # img_io[i, j, 0] = int(img_io[i, j, 0]*bright)
        # img_io[i, j, 1] = int(img_io[i, j, 1]*bright)
        # img_io[i, j, 2] = int(img_io[i, j, 2]*bright)
        for r in range(3): img_io[i, j, r] = color[r]
        
    
    save_folder = os.path.join(output_path,f"{strategy}_{img_id}_{mark}")
    
    os.makedirs(save_folder,exist_ok=True)
    save_path = os.path.join(save_folder, f"epoch_{epoch}_{cur_time}.png")
    imageio.imwrite(save_path, img_io)

if __name__ == "__main__":
    mark = gen_random_str()
    for epoch in [50,100,200,250,300,400, 500,1000,1500,2000,2500,3000,3500, 4000,4500,5000]:
        random_sample(input_img_path, output_path, epoch,mark)






