import time
import numpy as np
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import math




def print_list(in_list):
    line=''
    for item in in_list:
        line=line+str(item)+' '
    return line[0:-1]

def compute_det_curve(target_scores, nontarget_scores):
    n_scores = len(target_scores) + len(nontarget_scores)
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(len(target_scores)), np.zeros(len(nontarget_scores))))
    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]
    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = len(nontarget_scores) - (np.arange(1, n_scores + 1) - tar_trial_sums)
    # value <=thresholds，missing report
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / len(target_scores)))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / len(nontarget_scores)))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores
    return frr, far, thresholds

#target_scores,positive sample
def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    #draw table Far
    # eer
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    # FRR@FAR10% (%) approximate value
    abs_diffs_10 = np.abs(far - 0.1)
    min_index_10 = np.argmin(abs_diffs_10)
    frr_10 = round(np.mean(frr[min_index_10]),2)
    return eer, thresholds[min_index],frr_10,far, thresholds

# draw distribute table params, x num label,max_x
graph_group_num = 20
margin = 1
def compute_far(far_list,thresholds_list,group_num):
    # compute_far for FAR curse
    step=round(margin/group_num,4)
    sample_far_list=[]
    sample_thresholds_list=[]
    for i in range(0,group_num):
        # approximate value
        x=step/2+i*step
        temp_list=[abs(k-x) for k in thresholds_list]
        min_index = np.argmin(temp_list)
        sample_thresholds_list.append(round(x,4))
        sample_far_list.append(round(far_list[min_index],4))
    return sample_thresholds_list,sample_far_list

def compute_auc(real_distance_list,fake_distance_list):
    real_length = len(real_distance_list)
    fake_length = len(fake_distance_list)
    # print("real_len", real_length)
    # print("fake_len", fake_length)
    all_distances = np.concatenate((real_distance_list, fake_distance_list))
    labels = np.concatenate((np.ones(real_length), np.zeros(fake_length)))
    auc = roc_auc_score(labels, all_distances)
    return auc

#0-1化处理,阈值也要进行01化处理
def compute_distribution(real_distance_list,fake_distance_list,group_num):
    #分成
    step=round(margin/group_num,2)
    real_group_list=[0]*group_num
    fake_group_list=[0]*group_num
    for item in real_distance_list:
        i=int(item//step)
        real_group_list[i]=real_group_list[i]+1
    for item in fake_distance_list:
        i = int(item // step)
        fake_group_list[i]=fake_group_list[i]+1
    real_group_list=[round(i/len(real_distance_list),2) for i in real_group_list]
    fake_group_list=[round(i/len(fake_distance_list),2) for i in fake_group_list]
    return real_group_list,fake_group_list

def calculate_metrics(real_clip_distance_map,fake_clip_distance_map,label,iteration,evaluate_path):
    real_clip_distance_list = []
    fake_clip_distance_list = []
    # mean
    real_video_distance_list = []
    fake_video_distance_list = []
    # video metrics
    real_video_distance_map = defaultdict(lambda: [])
    fake_video_distance_map = defaultdict(lambda: [])
    for key, value in real_clip_distance_map.items():
        value=1.0/(np.mean(value)+0.1)
        real_clip_distance_list.append(value)
        # real_fadg0_sa1_SH_num
        part_list=key.split('_')
        video_key=part_list[0]+'_'+part_list[1]+'_'+part_list[2]
        if video_key in real_video_distance_map.keys():
            real_video_distance_map[video_key].append(value)
        else:
            real_video_distance_map[video_key]=[value]

    for key, value in fake_clip_distance_map.items():
        value=1.0/(np.mean(value)+0.1)
        fake_clip_distance_list.append(value)
        part_list = key.split('_')
        video_key = part_list[0] + '_' + part_list[1] + '_' + part_list[2]
        if video_key in fake_video_distance_map.keys():
            fake_video_distance_map[video_key].append(value)
        else:
            fake_video_distance_map[video_key] = [value]

    for _,value in real_video_distance_map.items():
        real_video_distance_list.append(np.mean(value))
    for _, value in fake_video_distance_map.items():
        fake_video_distance_list.append(np.mean(value))
    # standardization
    real_clip_distance_list = [math.log(i) for i in real_clip_distance_list]
    fake_clip_distance_list = [math.log(i) for i in fake_clip_distance_list]
    real_video_distance_list = [math.log(i)  for i in real_video_distance_list]
    fake_video_distance_list = [math.log(i) for i in fake_video_distance_list]
    clip_min_point=min(min(real_clip_distance_list),min(fake_clip_distance_list))
    clip_max_point=max(max(real_clip_distance_list),max(fake_clip_distance_list))+0.0001
    video_min_point = min(min(real_video_distance_list), min(fake_video_distance_list))
    video_max_point = max(max(real_video_distance_list), max(fake_video_distance_list))+0.0001

    real_clip_distance_list=[margin*(i-clip_min_point)/(clip_max_point-clip_min_point) for i in real_clip_distance_list]
    fake_clip_distance_list=[margin*(i-clip_min_point)/(clip_max_point-clip_min_point) for i in fake_clip_distance_list]
    real_video_distance_list = [margin*(i - video_min_point) / (video_max_point - video_min_point) for i in real_video_distance_list]
    fake_video_distance_list = [margin*(i - video_min_point) / (video_max_point - video_min_point) for i in fake_video_distance_list]

    # ROC video
    clip_auc  = compute_auc(real_clip_distance_list, fake_clip_distance_list)
    clip_eer, clip_thresholds,clip_frr_10,clip_far_list, clip_thresholds_list = compute_eer(real_clip_distance_list, fake_clip_distance_list)
    sample_clip_thresholds_list, sample_clip_far_list = compute_far(clip_far_list, clip_thresholds_list, graph_group_num)
    real_clip_group_list,fake_clip_group_list=compute_distribution(real_clip_distance_list, fake_clip_distance_list,graph_group_num)
    video_auc  = compute_auc(real_video_distance_list, fake_video_distance_list)
    video_eer, video_thresholds,video_frr_10,video_far_list, video_thresholds_list = compute_eer(real_video_distance_list, fake_video_distance_list)
    sample_video_thresholds_list, sample_video_far_list = compute_far(video_far_list, video_thresholds_list, graph_group_num)
    real_video_group_list, fake_video_group_list = compute_distribution(real_video_distance_list, fake_video_distance_list, graph_group_num)
    print('clip_AUC:{},clip_eer:{},clip_thresholds:{},clip_frr_10:{}'.format(clip_auc,clip_eer,clip_thresholds,clip_frr_10))
    print('video_AUC:{},video_eer:{},video_thresholds:{},video_frr_10:{}'.format(video_auc,video_eer,video_thresholds,video_frr_10))
    real_clip_group_list_line='real_clip_group_list:'+print_list(real_clip_group_list)
    fake_clip_group_list_line='fake_clip_group_list:'+print_list(fake_clip_group_list)
    real_video_group_list_line='real_video_group_list:'+print_list(real_video_group_list)
    fake_video_group_list_line='fake_video_group_list:'+print_list(fake_video_group_list)

    sample_clip_thresholds_list_line='sample_clip_thresholds_list:'+print_list(sample_clip_thresholds_list)
    sample_clip_far_list_line='sample_clip_far_list :'+print_list(sample_clip_far_list)
    sample_video_thresholds_list_line='sample_video_thresholds_list :'+print_list(sample_video_thresholds_list)
    sample_video_far_list_line='sample_video_far_list :'+print_list(sample_video_far_list)

    print(real_clip_group_list_line)
    print(fake_clip_group_list_line)
    print(real_video_group_list_line)
    print(fake_video_group_list_line)
    print(sample_clip_thresholds_list_line)
    print(sample_clip_far_list_line)
    print(sample_video_thresholds_list_line)
    print(sample_video_far_list_line)

    # record train and infer line
    with open(evaluate_path, 'a') as model_f:
        line0 = '\n'+label+':'+time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))+' model_'+str(iteration)+ '\n'
        line1 = 'clip_auc:' + ' ' + str(clip_auc) + '\n'
        line2 = 'clip_eer:' + ' ' + str(clip_eer) + '  '+'clip_thresholds:'+str(clip_thresholds)+'  '+'clip_frr_10:'+str(clip_frr_10)+ '\n'
        line3 = 'video_auc:' + ' ' + str(video_auc) + '\n'
        line4 = 'video_eer:' + ' ' + str(video_eer) + '  ' + 'video_thresholds:' + str(video_thresholds)+'  '+'video_frr_10:'+str(video_frr_10)+ '\n'
        model_f.write(line0)
        model_f.write(line1)
        model_f.write(line2)
        model_f.write(line3)
        model_f.write(line4)
        model_f.write(real_clip_group_list_line+'\n')
        model_f.write(fake_clip_group_list_line+'\n')
        model_f.write(real_video_group_list_line+'\n')
        model_f.write(fake_video_group_list_line+'\n')

        model_f.write(sample_clip_thresholds_list_line + '\n')
        model_f.write(sample_clip_far_list_line + '\n')
        model_f.write(sample_video_thresholds_list_line + '\n')
        model_f.write(sample_video_far_list_line + '\n')
        meta={
            'clip_auc':clip_auc,
            'clip_eer':clip_eer,
            'clip_thresholds':clip_thresholds,
            'clip_frr_10':clip_frr_10,
            'video_auc': video_auc,
            'video_eer': video_eer,
            'video_thresholds': video_thresholds,
            'video_frr_10': video_frr_10,
        }
    return meta


def test_auc():
    real_clip_distance_map = {
        'real_fadg0_sa1_0':1,
        'real_fadg0_sa2_1':1,
        'real_fadg0_sa3_2':3,
        'real_fadg0_sa1_5':4,
    }
    fake_clip_distance_map = {
        'fake_fadg0_sa4_0': 11,
        'fake_fmdg0_sa1_1': 7,
        'fake_fkdg0_sa2_1': 32,
        'fake_fkdu0_sa8_1': 9,
    }
    calculate_metrics(real_clip_distance_map,fake_clip_distance_map,'real',9,'path')


# def test_table():
#     real_distance_list=[1,2,3,4,5]
#     fake_distance_list=[6,7,8,9,10]
#     a,b=compute_table(real_distance_list, fake_distance_list,10)
#     print('hello')
#test_auc()

