import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np



sns.set_theme(style='darkgrid')

assets_path = "/home/ubuntu/projects/coding4paper/projects/libinr/assets"
out_dir = os.path.join(assets_path, "interval_study")
os.makedirs(out_dir, exist_ok=True)

BLUE = "#2E7EBB"
PRUPLE = "#7262AC"

ratio = 0.5
# 0.1
input_path_list = [
    f"/home/ubuntu/projects/coding4paper/projects/libinr/log/014_measurement_supp/02_1_2024-11-21-15:48:42_rnfw/mdd_data/{ratio}_1.pt",
    f"/home/ubuntu/projects/coding4paper/projects/libinr/log/014_measurement_supp/02_25_2024-11-21-15:48:42_kmch/mdd_data/{ratio}_25.pt",
    f"/home/ubuntu/projects/coding4paper/projects/libinr/log/014_measurement_supp/02_50_2024-11-21-15:48:42_kxne/mdd_data/{ratio}_50.pt",
]



for i,p in enumerate(input_path_list):
    data = torch.load(p)
    plt.figure(figsize=(10, 5))
    plt.plot(data.flatten(), linewidth=2.5, color=PRUPLE)
    plt.grid()
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.xlabel("Iteration",fontsize=25)
    plt.ylabel("Intersection Ratio",fontsize=25)
    plt.ylim(0,1)
    plt.grid()
    plt.savefig(os.path.join(out_dir,f"{i}_{ratio}.png"), bbox_inches='tight')