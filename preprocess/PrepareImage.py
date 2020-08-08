####划分视频数据集
import os
import cv2
import pickle
import numpy as np
image_attr_path=r'D:\document\pycharmproject\mouth_voice\output\image_attr.pkl'
image_feature_out_dir = r'D:\document\paper\personpaper\audio-visual_consistance\data\image_feature.pkl'
phoneme_video_model_file_path=r'D:\document\pycharmproject\mouth_voice\preprocess\phoneme_video_model_file.txt'
#目录结构 figure/word/image001.jpg
real_image_file_prefix=r'D:\document\paper\personpaper\audio-visual_consistance\data\timit_mouth'
high_image_file_prefix=r'D:\document\paper\personpaper\audio-visual_consistance\data\higher_quality_mouth'
low_image_file_prefix=r'D:\document\paper\personpaper\audio-visual_consistance\data\lower_quality_mouth'
test_set=['mwbt0','msjs1','mrgg0','mpgl0',
          'fram1','fjwb0','fjem0','felc0']

image_file_prefix_list=[real_image_file_prefix,high_image_file_prefix,low_image_file_prefix]

real_train_image_map={}
real_test_image_map={}
high_train_image_map={}
high_test_image_map={}
low_train_image_map={}
low_test_image_map={}
#for std
train_mean_high_image_list=[]
train_mean_low_image_list=[]
def divideImage():
    for image_file_prefix in image_file_prefix_list:
        figure_list=os.listdir(image_file_prefix)
        figure_list.sort()
        print(figure_list)

        file = open(phoneme_video_model_file_path)
        phoneme_sequence_num=0
        for line in file:
            # if phoneme_sequence_num==10:
            #     break
            # fadg0_sa1 SH 21 24 860.54 980.27 119.72
            line = line.strip('\n')
            info_list=line.split(" ")#fadg0_sa1
            figure_word=info_list[0].split("_")
            figure=figure_word[0]#fadg0

            word=figure_word[1]#sa1
            phoneme_label = info_list[1]#SH

            if 'timit_mouth' in image_file_prefix:
                phoneme_key_prefix = 'real'+'_'+info_list[0] + "_" + phoneme_label  # fadg0_sa1_SH
            else:
                phoneme_key_prefix = 'fake' + '_' + info_list[0] + "_" + phoneme_label  # fadg0_sa1_SH
            phoneme_key = phoneme_key_prefix + "_" + str(phoneme_sequence_num)
            phoneme_sequence_num += 1
            # 放在phoneme_sequence_num之后，保证与音频标号一致
            if figure not in figure_list:
                continue
            #计算的帧号从0开始，而源文件从1开始
            start_frame= int(info_list[2])+1
            end_frame  = int(info_list[3])+1
            image_group_list=[]
            # source fadg0_sa1_001.jpeg
            # image_group_list
            for i in range(start_frame ,end_frame+1):
                if len(str(i))==1:
                    image_no="00"+str(i)
                elif len(str(i))==2:
                    image_no = "0" + str(i)
                else:
                    image_no=str(i)
                source_image_suf=info_list[0]+"_"+image_no+".jpeg"
                source_image_path=os.path.join(image_file_prefix,figure,word,source_image_suf)

                frame = cv2.imread(source_image_path)
                #<class 'tuple'>: (40, 64)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                if figure not in test_set:
                    if image_file_prefix == real_image_file_prefix:
                        train_mean_high_image_list.append(frame_gray)
                        train_mean_low_image_list.append(frame_gray)
                    if image_file_prefix == high_image_file_prefix:
                        train_mean_high_image_list.append(frame_gray)
                    if image_file_prefix == low_image_file_prefix:
                        train_mean_low_image_list.append(frame_gray)
                image_group_list.append(frame_gray)

            if figure in test_set:
                #real_image_file_prefix, high_image_file_prefix, low_image_file_prefix
                if image_file_prefix==real_image_file_prefix:
                    real_test_image_map[phoneme_key]=image_group_list
                if image_file_prefix==high_image_file_prefix:
                    high_test_image_map[phoneme_key]=image_group_list
                if image_file_prefix==low_image_file_prefix:
                    low_test_image_map[phoneme_key]=image_group_list
            else:
                if image_file_prefix == real_image_file_prefix:
                    real_train_image_map[phoneme_key] = image_group_list
                if image_file_prefix == high_image_file_prefix:
                    high_train_image_map[phoneme_key] = image_group_list
                if image_file_prefix == low_image_file_prefix:
                    low_train_image_map[phoneme_key] = image_group_list

    high_mean = np.mean(train_mean_high_image_list, 0)
    high_std = np.std(train_mean_high_image_list, 0)
    low_mean = np.mean(train_mean_low_image_list, 0)
    low_std = np.std(train_mean_low_image_list, 0)
    image_attr={'high_mean':high_mean,
                'high_std':high_std,
                'low_mean':low_mean,
                'low_std':low_std}
    with open(os.path.join(image_attr_path), 'wb') as f:
        pickle.dump(image_attr, f)
    #combine
    train_high_map={**real_train_image_map,**high_train_image_map}
    train_low_map={**real_train_image_map,**low_train_image_map}
    test_high_map = {**real_test_image_map, **high_test_image_map}
    test_low_map = {**real_test_image_map, **low_test_image_map}
    image_feature={'train_high_map': train_high_map, 'train_low_map': train_low_map,
                   'test_high_map': test_high_map, 'test_low_map': test_low_map}
    with open(image_feature_out_dir, 'wb') as f:
        pickle.dump(image_feature, f)



def combine_audiovideofeature():
    image_attr_path = r'D:\document\pycharmproject\mouth_voice\output\image_attr.pkl'
    audio_feature_path=r'D:\document\paper\personpaper\audio-visual_consistance\data\audio_feature.pkl'
    image_feature_path=r'D:\document\paper\personpaper\audio-visual_consistance\data\image_feature.pkl'
    with open(image_attr_path, 'rb') as f:
        image_attr=pickle.load(f)
    with open(audio_feature_path, 'rb') as f:
        audio_feature=pickle.load(f)
    with open(image_feature_path, 'rb') as f:
        image_feature = pickle.load(f)
    train_high_list = []
    train_low_list = []
    test_high_list = []
    test_low_list = []
    train_high_map = image_feature['train_high_map']
    train_low_map = image_feature['train_low_map']
    test_high_map = image_feature['test_high_map']
    test_low_map = image_feature['test_low_map']
    audio_feature_map = audio_feature['audio_feature']

    #train_high
    for k,v in train_high_map.items():
        audio_k=k[5:]
        audio_feature=audio_feature_map[audio_k]
        assert len(audio_feature)==len(v)
        temp_list=[]
        for item in v:
            temp_list.append((item-image_attr['high_mean'])/image_attr['high_std'])
        list=[k,temp_list,audio_feature]
        train_high_list.append(list)
    #train_low
    for k,v in train_low_map.items():
        audio_k=k[5:]
        audio_feature=audio_feature_map[audio_k]
        assert len(audio_feature)==len(v)
        temp_list = []
        for item in v:
            temp_list.append((item - image_attr['low_mean']) / image_attr['low_std'])
        list=[k,temp_list,audio_feature]
        train_low_list.append(list)
    # test_high
    for k,v in test_high_map.items():
        audio_k=k[5:]
        audio_feature=audio_feature_map[audio_k]
        assert len(audio_feature)==len(v)
        temp_list = []
        for item in v:
            temp_list.append((item - image_attr['high_mean']) / image_attr['high_std'])
        list=[k,temp_list,audio_feature]
        test_high_list.append(list)
    # test_low
    for k,v in test_low_map.items():
        audio_k=k[5:]
        audio_feature=audio_feature_map[audio_k]
        assert len(audio_feature)==len(v)
        temp_list = []
        for item in v:
            temp_list.append((item - image_attr['low_mean']) / image_attr['low_std'])
        list = [k, temp_list, audio_feature]
        test_low_list.append(list)

    train_root=r'D:\document\paper\personpaper\audio-visual_consistance\data\train'
    test_root=r'D:\document\paper\personpaper\audio-visual_consistance\data\test'

    train_high={'train_high':train_high_list}
    with open(os.path.join(train_root, 'train_high.pkl'), 'wb') as f:
        pickle.dump(train_high, f)

    train_low = {'train_low': train_low_list}
    with open(os.path.join(train_root, 'train_low.pkl'), 'wb') as f:
        pickle.dump(train_low, f)

    test_high = {'test_high': test_high_list}
    with open(os.path.join(test_root, 'test_high.pkl'), 'wb') as f:
        pickle.dump(test_high, f)

    test_low = {'test_low': test_low_list}
    with open(os.path.join(test_root, 'test_low.pkl'), 'wb') as f:
        pickle.dump(test_low, f)

#divideImage()
print('*')
combine_audiovideofeature()