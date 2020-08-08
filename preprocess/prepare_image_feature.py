import os
import cv2
import pickle
import numpy as np
import sys

test_set=['mwbt0','msjs1','mrgg0','mpgl0',
          'fram1','fjwb0','fjem0','felc0']

def divideImage(image_file_root_list,phoneme_info_path,audio_feature_path,out_dir):
    train_mean_high_image_list=[]
    train_mean_low_image_list=[]

    # map
    real_train_image_map = {}
    real_test_image_map={}
    high_train_image_map = {}
    high_test_image_map = {}
    low_train_image_map = {}
    low_test_image_map = {}

    for image_file_root in image_file_root_list:
        figure_ids=os.listdir(image_file_root).sort()
        print(figure_ids)

        with open(phoneme_info_path,'r') as phoneme_info_file:
            #fadg0_sa1 SH 21 24 860.54 980.27 119.72789115700004
            for i,line in enumerate(phoneme_info_file):
                phoneme_info = line.strip().split()
                figure_id = phoneme_info[0].split("_")[0]
                # figure_id from phoneme_info_path,sometines not in fake figures
                if figure_id not in figure_ids:
                    continue
                word_id = phoneme_info[0].split("_")[1]
                phoneme_label = phoneme_info[1]

                # real_fadg0_sa1_SH_i
                if 'timit_mouth' in image_file_root:
                    phoneme_unit_label = 'real_'+phoneme_info[0] + "_" + phoneme_label + '_' + str(i)
                else:
                    phoneme_unit_label = 'fake_' + phoneme_info[0] + "_" + phoneme_label + '_' + str(i)

                #video frame start from 1
                start_frame = int(phoneme_info[2])+1
                end_frame = int(phoneme_info[3])+1
                phoneme_unit_list=[]
                for i in range(start_frame, end_frame + 1):
                    image_number=str(i).rjust(3,'0')
                    # fadg0_sa1_001.jpeg
                    mouth_image_suf = phoneme_info[0] + "_" + image_number + ".jpeg"
                    mouth_image_path = os.path.join(image_file_root, figure_id, word_id, mouth_image_suf)
                    frame = cv2.imread(mouth_image_path)
                    # <class 'tuple'>: (40, 64)
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    # calculate for mean and std

                    if figure_id not in test_set:
                        # real image
                        if image_file_root == image_file_root_list[0]:
                            train_mean_high_image_list.append(frame_gray)
                            train_mean_low_image_list.append(frame_gray)
                        # high fake image
                        elif image_file_root == image_file_root_list[1]:
                            train_mean_high_image_list.append(frame_gray)
                        # low fake image
                        else:
                            train_mean_low_image_list.append(frame_gray)
                phoneme_unit_list.append(frame_gray)
                # train set or fake set
                if figure_id in test_set:
                    # real image
                    if image_file_root == image_file_root_list[0]:
                        real_test_image_map[phoneme_unit_label]=phoneme_unit_list
                    # high fake image
                    elif image_file_root == image_file_root_list[1]:
                        high_test_image_map[phoneme_unit_label]=phoneme_unit_list
                    # low fake image
                    else:
                        low_test_image_map[phoneme_unit_label]=phoneme_unit_list
                else:
                    if image_file_root == image_file_root_list[0]:
                        train_mean_high_image_list.append(phoneme_unit_list)
                        train_mean_low_image_list.append(phoneme_unit_list)
                        real_train_image_map[phoneme_unit_label] = phoneme_unit_list
                    # high fake image
                    elif image_file_root == image_file_root_list[1]:
                        train_mean_high_image_list.append(phoneme_unit_list)
                        high_train_image_map[phoneme_unit_label] = phoneme_unit_list
                    # low fake image
                    else:
                        train_mean_low_image_list.append(phoneme_unit_list)
                        low_train_image_map[phoneme_unit_label] = phoneme_unit_list
            # mean and std
            high_mean = np.mean(train_mean_high_image_list, 0)
            high_std = np.std(train_mean_high_image_list, 0)
            low_mean = np.mean(train_mean_low_image_list, 0)
            low_std = np.std(train_mean_low_image_list, 0)
            image_attr = {'high_mean': high_mean,
                          'high_std': high_std,
                          'low_mean': low_mean,
                          'low_std': low_std}
            with open(os.path.join(out_dir,'image_attr.pkl'), 'wb') as f:
                pickle.dump(image_attr, f)

            train_high_map = {**real_train_image_map, **high_train_image_map}
            train_low_map = {**real_train_image_map, **low_train_image_map}
            test_high_map = {**real_test_image_map, **high_test_image_map}
            test_low_map = {**real_test_image_map, **low_test_image_map}

            with open(audio_feature_path, 'rb') as f:
                audio_feature = pickle.load(f)
            audio_feature_map = audio_feature['audio_feature']

            for data_label, data_map in zip(['train_high', 'train_low', 'test_high', 'test_low'],\
                                       [train_high_map, train_low_map, test_high_map,test_low_map]):
                for label, values in data_map.items():
                    # high or low
                    quality=data_label.split('_')[1]
                    image_mean=image_attr[quality+'_mean']
                    image_std=image_attr[quality+'_std']
                    image_features=[(value-image_mean)/image_std for value in values]
                    # remove 'fake_' or 'real_'
                    audio_label = label[5:]
                    audio_features = audio_feature_map[audio_label]
                    assert len(audio_features) == len(image_features)

                    audio_visual_features = [label, image_features, audio_features]
                    data_map[label] = audio_visual_features
                norm_data = {data_label: data_map}
                with open(os.path.join(out_dir, data_label+'.pkl'), 'wb') as f:
                    pickle.dump(norm_data, f)

if __name__=='__main__':
    # # audio data
    # real_image_dir = sys.argv[1]
    # high_fake_image_dir = sys.argv[2]
    # low_fake_image_dir = sys.argv[3]
    # phoneme_info_path = sys.argv[4]

    real_image_dir = r'D:\document\paper\personpaper\audio-visual_consistance\data\timit_mouth'
    high_image_dir = r'D:\document\paper\personpaper\audio-visual_consistance\data\higher_quality_mouth'
    low_image_dir = r'D:\document\paper\personpaper\audio-visual_consistance\data\lower_quality_mouth'

    image_file_root_list = [real_image_dir,high_image_dir,low_image_dir]
    phoneme_info_path = r'D:\document\pycharmproject\AVCDetection\preprocess\phoneme_video_model_file.txt'
    audio_feature_path=r''
    out_dir = r'D:\document\pycharmproject\AVCDetection\output'

    divideImage(image_file_root_list, phoneme_info_path,audio_feature_path,out_dir)

