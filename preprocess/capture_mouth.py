import cv2
import os
import yaml
from argparse import ArgumentParser

def capture_mouth_region(data_dir,output_dir):
    for root,dirs,files in os.walk(data_dir,):
        for file in files:
            #获取word_name
            word=os.path.basename(root)
            root_end_with_video=os.path.dirname(root)
            root_end_with_figure=os.path.dirname(root_end_with_video)
            figure=os.path.basename(root_end_with_figure)
            if word not in ['head','head2','head3'] and os.path.splitext(file)[1] == '.jpeg':
                mouth_dir=os.path.join(output_dir,figure,word)
                print('mouth_dir:',mouth_dir)
                if not os.path.exists(mouth_dir):
                    cmd = 'mkdir -p  ' + mouth_dir
                    print(cmd)
                    os.system(cmd)
                mouth_file=os.path.join(mouth_dir,figure+"_"+word+"_"+file)
                print('mouth_file:', mouth_file)
                # capture mouth
                file_path=os.path.join(root,file)
                frame = cv2.imread(file_path)
                x, y, width, height = face_detect.face_detect(frame)
                w = width + x
                h = height + y
                mouth_frame = frame[y:h, x:w]
                gray_image = cv2.cvtColor(mouth_frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(mouth_file, gray_image)

def face_detect(color_image,model_path):
    # color to gray
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    # face detector
    detector = dlib.get_frontal_face_detector()
    # feature point detector
    predictor = dlib.shape_predictor(model_path)
    # The 1 in the second argument indicates that we should upsample the image 1 time.
    faces = detector(gray_image, 1)
    for face in faces:
        # 68 landmarks of the human face
        shape = predictor(color_image, face)
        point = []
        for idx, pt in enumerate(shape.parts(), 1):
            if idx == 67:
                print("location:", pt.x, pt.y)
                point.append(pt)
        # 计算矩形点，长和宽
        width = 64
        height = 40
        #  top left corner
        x = int(point[0].x - width / 2)
        y = int(point[0].y - height / 2 - 5)
        return x, y, width, height

if __name__=='__main__':

    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config.yaml')
    args = parser.parse_args()

    #args.config = r'preprocess\preprocess_config.yaml'

    # load config file
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data_dir = config['origi_data_dir']
    output_dir = config['mouth_output_dir']
    face_model_path = config['face_model_path']
    capture_mouth_region(data_dir,output_dir,face_model_path)

