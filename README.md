# Audio-visual Consistency Detection(AVCD)
This is the implementation of the paper **Deepfake Video Detection Using Audio-Visual Consistency**, in which a  method is employed to Deepfake detection by exploiting audio-visual consistency. The project proposes an audio-visual coupling model(**AVCM**) for Audio-Visual Consistency Detection(**AVCD**). AVCM is employed to measure similarity of audio-visual pairs. The similarity is a metric used to indicate the degrees of synchronization of audio-visual pairs. Ideally, synchronized pairs correspond to high scores and asynchronous pairs opposite. The model is trained to  expand the similarity of positive samples and minify the negative ones, which eventually forms a threshold to  distinguish samples.

# Dependencies
 * python        3.6+
 * pytorch       1.0.1
 * SoundFile     0.10.2
 * librosa       0.7.2
 * opencv-python 4.3.0.36
 * numpy         1.19.1
 * dlib 19.7.0

# Preprocess
## dataset
We evaluate proposed method on [VidTIMIT](http://conradsanderson.id.au/vidtimit/#downloads) and [DeepfakeTIMIT](https://www.idiap.ch/dataset/deepfaketimit) where synchronous audio-visual pairs are produced from VidTIMIT and asynchronous pairs from DeepfakeTIMIT. Videos in DeepfakeTIMIT are derived from VidTIMIT with faces swapped using the open source GAN-based approach, and transcripts in speeches are from [TIMIT corups](http://academictorrents.com/details/34e2b78745138186976cbc27939b1b34d18bd5b3).
## phoneme alignment
We have audio aligned by using a phoneme-based alignment tool [P2FA](https://babel.ling.upenn.edu/phonetics/old_website_2015/p2fa/index.html). See this [blog](https://blog.csdn.net/jojozhangju/article/details/51951622) post for details on how to use it. The phoneme alignment information has provided [here](https://github.com/BrightGu/AVCDetection/blob/master/preprocess/phoneme_video_model_file.txt).
## audio features
We use mel-scale spectrograms as audio features. In our experiments, mel-scale spectrograms are computed from a power spectrum (power of magnitude of 2048-sized STFT) on 40-ms windows length which result in  512-dimensional vectors.
> python capture_mouth.py -c config.yaml
> python prepare_audio_feature.py -c config.yaml
## audio-visual features
In this section, we produce audio-visual pairs based on phoneme alignment infomation. Each mouth frame is related to a fixed-length audio segment, then sequential pairs with same phoneme labels are assembled into  specific phoneme units. The phoneme units are used for AVCM training.
> python prepare_image_feature.py -c config.yaml
# AVCM
We propose AVCM, which is a CNN model consists of audio architecture and video architecture. AVCM is employed to measure similarity of audio-visual pairs. The details of AVCM are described as follow.

![Image text](https://github.com/BrightGu/AVCDetection/blob/master/figure/AVCM%20architecture.png)

# train or infer
> python main.py -c config.yaml

