from  os.path import join,basename
import os
import pickle


file1 = r'D:\document\pycharmproject\AVCDetection\output\image_attr.pkl'
with open(file1,'rb') as f:
   data1=pickle.load(f)
print(data1)
file2 = r'D:\document\paper\personpaper\audio-visual_consistance\data\image_attr.pkl'
with open(file2,'rb') as f:
   data2=pickle.load(f)
print(data2)


print("hello")





