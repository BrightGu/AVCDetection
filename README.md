# Audio-visual Consistency Detection(AVCD)
This is the implementation of the paper **Deepfake Video Detection Using Audio-Visual Consistency**, in which a  method is employed to Deepfake detection by exploiting audio-visual consistency. The project proposes an audio-visual coupling model(**AVCM**) for Audio-Visual Consistency Detection(**AVCD**). AVCM is employed to measure similarity of audio-visual pairs. The similarity is a metric used to indicate the degrees of synchronization of audio-visual pairs. Ideally, synchronized pairs correspond to high scores and asynchronous pairs opposite. The model is trained to  expand the similarity of positive samples and minify the negative ones, which eventually forms a threshold to  distinguish samples.

# Dependencies
 * python        3.6+
 * pytorch       1.0.1
 * SoundFile     0.10.2
 * librosa       0.7.2
 * opencv-python 4.3.0.36
 * numpy         1.19.1

# Preprocess
## dataset
 * [VidTIMIT](http://conradsanderson.id.au/vidtimit/#downloads)
 * [DeepfakeTIMIT](https://www.idiap.ch/dataset/deepfaketimit)
## phoneme alignment
we have audio aligned by using a phoneme-based alignment tool [P2FA](https://babel.ling.upenn.edu/phonetics/old_website_2015/p2fa/index.html). See this [blog](https://blog.csdn.net/jojozhangju/article/details/51951622) post for details on how to use it. The phoneme_info.txt has provided here.
## audio features
> python prepare_audio_feature.py -c config.yaml
## audio-visual features
> python prepare_image_feature.py -c config.yaml
# AVCM

# train

# infer
