import scipy.io.wavfile
import os
import math
import sys
#from python_speech_features import base
import librosa
import numpy as np
import pickle
from preprocess.tacotron.hyperparams import Hyperparams as hy
from preprocess.tacotron import utils

# file_path=r'D:\document\pycharmproject\mouth_voice\preprocess\phoneme_video_model_file.txt'
# audio_file_prefix=r'D:\document\paper\personpaper\audio-visual_consistance\data\timit_audio2'
# mean_std_path=r'D:\document\pycharmproject\mouth_voice\output'
test_set=['mwbt0','msjs1','mrgg0','mpgl0',
          'fram1','fjwb0','fjem0','felc0']

# ./figure/word.wav
# 292k 32000 1ch 512kbps


# signal=[]
# sample_rate=32000

#total feature for mean and std



def get_audio_mel(data_dir,phoneme_info_path,out_dir):
    audio_signal=[]
    loaded_audio_list=[]
    audio_feature_map = {}
    train_audio_feature_list = []
    with open(phoneme_info_path,'r') as phoneme_info_file:
        #fadg0_sa1 SH 21 24 860.544 980.27 119.72
        for i, line in enumerate(phoneme_info_file):
            phoneme_info = line.strip().split()
            figure_id=phoneme_info[0].split("_")[0]
            word_id=phoneme_info[0].split("_")[1]
            phoneme_label=phoneme_info[1]
            # fadg0_sa1_SH_i
            phoneme_unit_label=phoneme_info[0]+"_"+phoneme_label+'_'+str(i)

            # The corresponding video frame
            start_frame = int(phoneme_info[2])
            end_frame = int(phoneme_info[3])
            # interval start and end time
            start_time = float(phoneme_info[4])
            end_time = float(phoneme_info[5])

            # audio tag
            audio_label=figure_id+'_'+word_id+'.wav'
            if audio_label not in loaded_audio_list:
                audio_file=os.path.join(data_dir,figure_id,word_id+'.wav')
                audio_signal, _ = librosa.load(audio_file, hy.sr)
                loaded_audio_list.append(audio_label)

            # clip segment with matching strategy

            phoneme_unit_mel_features=[]
            for index in range(start_frame, end_frame + 1):
                if index == start_frame:
                    start_point = math.floor(start_time * hy.sr * 1.0/1000 )
                    end_point = start_point + int(hy.frame_length * hy.sr * 1.0 )
                elif index == end_frame:
                    end_point = math.floor(end_time * hy.sr * 1.0 / 1000.0)
                    start_point = end_point - int(hy.frame_length * hy.sr * hy.sr * 1.0 )
                else:
                    start_point = math.floor(index*hy.frame_length * hy.sr * 1.0 )
                    end_point=start_point+int(hy.frame_length * hy.sr * 1.0/1000 )
                clip_signal = audio_signal[start_point:end_point]
                mel, _ = utils.get_spectrogramsfromsignal(clip_signal)
                # n_mels 512
                assert len(mel[0]) == hy.n_mels
                phoneme_unit_mel_features.append(mel[0])
                # calcute for mean and std
                if figure_id not in test_set:
                    train_audio_feature_list.append(mel[0])
            audio_feature_map[phoneme_unit_label]=phoneme_unit_mel_features

    train_audio_features = np.concatenate(train_audio_feature_list)
    print('len(train_audio_feature_list)', len(train_audio_feature_list))
    mean = np.mean(train_audio_features, axis=0)
    std = np.std(train_audio_features, axis=0)
    attr = {'mean': mean, 'std': std}
    with open(os.path.join(out_dir, 'audio_attr.pkl'), 'wb') as f:
        pickle.dump(attr, f)

    normalized_audio_feature_map = {}
    for key, val in audio_feature_map.items():
        processed_value_list=[(i - mean) / std for i in val]
        normalized_audio_feature_map[key]=processed_value_list
    # audio_feature.pkl
    audio_feature = {'audio_feature': normalized_audio_feature_map}
    with open(os.path.join(out_dir, 'audio_feature.pkl'), 'wb') as f:
        pickle.dump(audio_feature, f)


if  __name__  == '__main__':
    # # audio data
    # data_dir = sys.argv[1]
    # # phoneme info
    # phoneme_info_path = sys.argv[2]
    # # std mean feature
    # out_dir=sys.argv[3]

    data_dir=r'D:\document\paper\personpaper\audio-visual_consistance\data\timit_audio2'
    phoneme_info_path=r'D:\document\pycharmproject\AVCDetection\preprocess\phoneme_video_model_file.txt'
    out_dir=r'D:\document\pycharmproject\AVCDetection\output'


    get_audio_mel(data_dir,phoneme_info_path,out_dir)

