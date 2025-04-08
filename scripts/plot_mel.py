
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
 
FONT_SIZE = 16
plt.rcParams.update({'font.size': FONT_SIZE})
plt.rcParams['lines.linewidth'] = 2.5
# plt.figure(figsize=(10, 10))

assets_path = "/home/ubuntu/projects/coding4paper/projects/libinr/assets"
out_dir = os.path.join(assets_path, "audio_mel")
os.makedirs(out_dir, exist_ok=True)

idx = "07"
audio_data_full = f'/home/ubuntu/projects/coding4paper/projects/libinr/log/050_audio_sampler/full_2024-11-21-12:13:57_idqn/full_pred_audio/3000_{idx}.wav'
audio_data_evos = f'/home/ubuntu/projects/coding4paper/projects/libinr/log/050_audio_sampler/evos_2024-11-21-12:13:56_kqqf/full_pred_audio/5000_{idx}.wav'
audio_data_gt = f'/home/ubuntu/projects/coding4paper/data/libri_test_clean_121726/{idx}.flac'

# sample_rate = 16000

for i,audio_data in enumerate((audio_data_gt,audio_data_evos,audio_data_full)):

    x , sr = librosa.load(audio_data)
    # x = librosa.resample(x, orig_sr=sr, target_sr=sample_rate)
    # crop
    target_len = min(sr * 5, len(x))
    x = x[:target_len]
    print('x : ', len(x) )
 
    mel_spect = librosa.feature.melspectrogram(y=x, sr=sr )
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    mel_spect = mel_spect[10:,:]
    librosa.display.specshow(mel_spect, y_axis='mel', fmax=sr/2, x_axis='time')
    
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Fequency', fontsize=20)
    # plt.title('Mel Spectrogram')
    # plt.colorbar(format='%+2.0f dB')
    plt.savefig(os.path.join(out_dir,f"mel_{i}.png"),bbox_inches='tight')