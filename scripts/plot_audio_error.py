import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_theme(style='darkgrid')
import numpy as np

FONT_SIZE = 25
plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams['lines.linewidth'] = 2.5

GRAY = "#5F5F5F"
PRUPLE = "#7262AC"
RED = "#D92523"
RED_S = "#FCAB8F"
PRUPLE_S = "#cfcfe5"

# io
# input_path = "/home/ubuntu/projects/coding4paper/projects/libinr/log/050_audio_sampler/evos_2024-11-20-19:55:51_vovj/error_tensor_pts/error_10_5000.pt"
input_path = "/home/ubuntu/projects/coding4paper/projects/libinr/log/050_audio_sampler/full_2024-11-20-19:55:51_qqas/error_tensor_pts/error_10_3000.pt"

assets_path = "/home/ubuntu/projects/coding4paper/projects/libinr/assets"
out_dir = os.path.join(assets_path, "audio_error")
os.makedirs(out_dir, exist_ok=True)

file_name = f"error_full_10.png"
out_path = os.path.join(out_dir, file_name)

# read data
sr = 16000
error = torch.load(input_path)
time_axis = np.linspace(0, len(error) / sr, num=len(error))

plt.figure(figsize=(10, 5))
plt.plot(time_axis, error, color= RED)
plt.ylim(-1, 1) 
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Error of Reconstructed Audio")
# plt.legend()
plt.grid()
plt.savefig(out_path, bbox_inches='tight')
