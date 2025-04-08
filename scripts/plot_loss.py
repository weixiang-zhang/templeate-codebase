import json
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

FONT_SIZE = 26
plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams['lines.linewidth'] = 2.5




GRAY = "#5F5F5F"
GRAY_S = "#CECECE"
PRUPLE = "#7262AC"
PRUPLE_S = "#cfcfe5"
BLUE = "#2E7EBB"
BLUE_S = "#B7D4EA"
GREEN = "#2E974E"
GREEN_S = "#B8E3B2"
ORANGE = "#E25508"
ORANGE_S = "#FDC38D"
RED = "#D92523"
RED_S = "#FCAB8F"

KEY_MAPPER = {
    "full":("Standard", GRAY),
    "fm_cur":("Ours", PRUPLE),
    "nmt_incre_step":("NMT", BLUE),
    "soft" : ("Soft Mining", GREEN),
    "expansive":("Expansive",ORANGE),
    "egra":("EGRA",RED)
}

def smooth(y, window_size=5):
    y_smooth = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
    y_std = np.std(y) * np.ones_like(y_smooth)
    return y_smooth, y_std

def load_json(filepath):
    with open(filepath, 'r') as f:
        loaded_json = json.load(f)
    return loaded_json

def replot_loss_psnr():
    input_folder = "/home/ubuntu/projects/coding4paper/projects/libinr/log/000_visualized/dense_psnr"
    time_k = 0.6 # 系数 乘一下 | 97.8 / 165
    xs_dic = {}
    ys_dic = {}
    y_buffer_dic = {}
    # input_path_list = glob.glob(os.path.join(input_folder, '*.json'))
    input_path_list = [
        "/home/ubuntu/projects/coding4paper/projects/libinr/log/000_visualized/dense_psnr/full_2024-10-13-18_31_39.json",
        "/home/ubuntu/projects/coding4paper/projects/libinr/log/000_visualized/dense_psnr/egra_2024-10-13-18_31_39.json",
        "/home/ubuntu/projects/coding4paper/projects/libinr/log/000_visualized/dense_psnr/soft_2024-10-13-21_39_50.json",
        "/home/ubuntu/projects/coding4paper/projects/libinr/log/000_visualized/dense_psnr/nmt_incre_step_2024-10-13-18_31_39.json",
        "/home/ubuntu/projects/coding4paper/projects/libinr/log/000_visualized/dense_psnr/expansive_2024-10-13-18_31_39.json",
        "/home/ubuntu/projects/coding4paper/projects/libinr/log/000_visualized/dense_psnr/fm_cur_2024-10-13-23_13_05.json",
    ]

    min_time = 1000
    for input_path in input_path_list:
        key= os.path.splitext(os.path.basename(input_path))[0].split("_2024")[0]
        print('key: ', key)
        json_data = load_json(input_path)
        xs = []
        ys = []
        y_buffers = []
        _start_time = json_data[0][0]
        _end_time = json_data[-1][0]
        min_time = min(_end_time - _start_time, min_time)

        y_raw_list = []
        
        for _time, _epoch, _data in json_data:
            scaled_time = time_k * (_time - _start_time)
            y_raw_list.append(_data)
            # smoothed, std = smooth(np.array(_data))
            xs.append(scaled_time)
            # ys.append(_data)
            # ys.append(smoothed)
            # y_buffers.append(std)
        
    
        smoothed, std = smooth(np.array(y_raw_list), window_size=10)
        _len = smoothed.shape[0]

        xs_dic[key] = xs[:_len]
        ys_dic[key] = smoothed
        y_buffer_dic[key] = std * 0.1

    plt.figure(figsize=(10, 10))
    
    plt.xlabel("Training Time (sec)") 
    
    
    
    plt.grid(True)

    
    plt.title('PSNR / Training Time')
    plt.ylabel('Reconstructed PSNR (dB)')
   
    


    for key in list(xs_dic.keys()):
        if key in ["nmt_incre", "random", "nmt_dense"]:
            continue
        _x = np.array(xs_dic[key])
        _y = np.array(ys_dic[key])
        _std = np.array(y_buffer_dic[key])
        plt.plot(_x, _y, label=KEY_MAPPER[key][0], color = KEY_MAPPER[key][1])
        plt.fill_between(_x, _y - _std, _y + _std, alpha=0.1, color = KEY_MAPPER[key][1])

    plt.xlim(0,90)
    plt.ylim(25, 38) 
    
    

    # legend_labels = ["Standard","NMT", "EGRA","Soft Mining", "Expansive","Ours"]
    plt.legend(loc='lower right',fontsize=FONT_SIZE)



    plt.savefig(os.path.join(input_folder, "psnr_out.png"), bbox_inches='tight') 


def replot_loss_ssim():
    input_folder = "/home/ubuntu/projects/coding4paper/projects/libinr/log/000_visualized/dense_ssim"
    time_k = 0.6 # 系数 乘一下 | 97.8 / 165
    xs_dic = {}
    ys_dic = {}
    y_buffer_dic = {}
    # input_path_list = glob.glob(os.path.join(input_folder, '*.json'))
    input_path_list = [
        "/home/ubuntu/projects/coding4paper/projects/libinr/log/000_visualized/dense_ssim/full_2024-10-13-18_31_39 (1).json",
        "/home/ubuntu/projects/coding4paper/projects/libinr/log/000_visualized/dense_ssim/egra_2024-10-13-18_31_39 (1).json",
        "/home/ubuntu/projects/coding4paper/projects/libinr/log/000_visualized/dense_ssim/soft_2024-10-13-21_39_50 (2).json",
        "/home/ubuntu/projects/coding4paper/projects/libinr/log/000_visualized/dense_ssim/nmt_incre_step_2024-10-13-18_31_39 (1).json",
        "/home/ubuntu/projects/coding4paper/projects/libinr/log/000_visualized/dense_ssim/expansive_2024-10-13-18_31_39 (1).json",
        "/home/ubuntu/projects/coding4paper/projects/libinr/log/000_visualized/dense_ssim/fm_cur_2024-10-13-23_13_05 (1).json",

    ]

    min_time = 1000
    for input_path in input_path_list:
        key= os.path.splitext(os.path.basename(input_path))[0].split("_2024")[0]
        print('key: ', key)
        json_data = load_json(input_path)
        xs = []
        ys = []
        y_buffers = []
        _start_time = json_data[0][0]
        _end_time = json_data[-1][0]
        min_time = min(_end_time - _start_time, min_time)

        y_raw_list = []
        
        for _time, _epoch, _data in json_data:
            scaled_time = time_k * (_time - _start_time)
            y_raw_list.append(_data)
            # smoothed, std = smooth(np.array(_data))
            xs.append(scaled_time)
            # ys.append(_data)
            # ys.append(smoothed)
            # y_buffers.append(std)
        
    
        smoothed, std = smooth(np.array(y_raw_list), window_size=5)
        _len = smoothed.shape[0]

        xs_dic[key] = xs[:_len]
        ys_dic[key] = smoothed
        y_buffer_dic[key] = std * 0.1

    plt.figure(figsize=(10, 10))
    
    plt.xlabel("Training Time (sec)") 
    
    
    
    plt.grid(True)

    
    plt.title('SSIM / Training Time')
    plt.ylabel('Reconstructed SSIM')
   

    for key in list(xs_dic.keys()):
        if key in ["nmt_incre", "random", "nmt_dense"]:
            continue
        _x = np.array(xs_dic[key])
        _y = np.array(ys_dic[key])
        _std = np.array(y_buffer_dic[key])
        plt.plot(_x, _y, label=KEY_MAPPER[key][0], color = KEY_MAPPER[key][1])
        plt.fill_between(_x, _y - _std, _y + _std, alpha=0.1, color = KEY_MAPPER[key][1])

    plt.xlim(0,90)
    plt.ylim(0.6, 1) 
    
    

    # legend_labels = ["Standard","NMT", "EGRA","Soft Mining", "Expansive","Ours"]
    plt.legend(loc='lower right',fontsize=FONT_SIZE)



    plt.savefig(os.path.join(input_folder, "ssim_out.png"), bbox_inches='tight') 


if __name__ == "__main__":
    # psnr
    
    #ssim
    # input_folder = "/home/ubuntu/projects/coding4paper/projects/libinr/log/000_visualized/dense_ssim"
    replot_loss_psnr()
    replot_loss_ssim()


