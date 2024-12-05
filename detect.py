import os
import sys
import json
import torch
import config
import logging
from PIL import Image
from shutil import copy2
from datetime import datetime
from dataset import get_transforms
from efficientnet_pytorch.model import EfficientNet
from preprocessor.image_preprocess import auto_adjustments, opencv_processing
from preprocessor.image_crop import image_crop
from config import PATH, MODEL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Detect:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.model_name = MODEL['MODEL_NAME']
        self.weights_dir = PATH['WEIGHTS_PATH']
        self.labels_file = PATH['LABELS']
        self.model_weights_file = os.path.join(self.weights_dir, self.model_name + '.pth')
        self.image_size = EfficientNet.get_image_size(self.model_name)
        self.tfms = get_transforms(self.image_size)

    def run_detect(self):
        self.labels_map = json.load(open(self.labels_file))
        self.labels_map = [self.labels_map[str(i)] for i in range(len(self.labels_map))]

        # 학습 모델 호출
        if device.type == 'cpu':
            model = EfficientNet.from_pretrained(self.model_name, num_classes=len(self.labels_map))
            model.load_state_dict(torch.load(self.model_weights_file, map_location=device))
            
        else:
            model = EfficientNet.from_pretrained(self.model_name, 
                                                 weights_path=self.model_weights_file,
                                                 num_classes=len(self.labels_map))

        model.to(device)
        model.eval()

        # output 경로에 존재하지 않는 이미지 리스트만 img_files 변수에 할당
        img_list = set()

        for sub_dir in ['OK', 'PASS', 'NG']:
            sub_path = os.path.join(self.output_dir, sub_dir)
            os.makedirs(sub_path, exist_ok=True)

            if os.path.exists(sub_path):
                img_list.update(os.listdir(sub_path))
        
        # 이미지 파일 리스트 : ['MFR01328AE;2408221189A.png', 'MFR01327AE;2409110702A.png', ....]
        imagelist, filenamelist = image_crop(self.input_dir) # 크롭된 PIL 이미지, 파일명 호출
        img_files = [f for f in filenamelist if f not in img_list] 

        for img_file, img in zip(img_files, imagelist):
            try:
                img_path = os.path.join(self.input_dir, img_file)

                # 이미지 전처리
                img_processed = auto_adjustments(img) # 자동 밝기, 대비, 선명도 조정
                img_processed = opencv_processing(img_processed) # 노이즈 제거
                
                img_processed = img_processed.convert('L') # Grayscale 변환
                img_processed = img_processed.convert('RGB') # 3차원 변환

                img_processed = self.tfms(img_processed).unsqueeze(0)
                img_processed = img_processed.to(device)

                # 이미지 예측
                with torch.no_grad():
                    logits = model(img_processed)
                    pred = torch.argmax(logits, dim=1).item()
                    prob = torch.softmax(logits, dim=1)[0, pred].item()
                    label = self.labels_map[pred]
                
                logging.info('-' * 100)
                logging.info(f'Image File : {img_file}')
                logging.info('{:<75} ({:.2f}%)'.format(label, prob*100))
                logging.info('-' * 100)

                ok_dir = os.path.join(self.output_dir, 'OK')
                pass_dir = os.path.join(self.output_dir, 'PASS')
                ng_dir = os.path.join(self.output_dir, 'NG')

                # 예측 결과 저장
                if label == 'OK':
                    copy2(img_path, ok_dir)
                elif label == 'PASS':
                    copy2(img_path, pass_dir)
                elif label == 'NG':
                    copy2(img_path, ng_dir)

            except Exception as e:
                logging.info(f'{img_file} : {e}')

if __name__ == '__main__':
    try:
        path = sys.argv[1]
        config.set_config(path)
        input_dir = sys.argv[2]
        output_dir = sys.argv[3]
    
        if not os.path.exists('log'):
            os.makedirs('log')

        logging.basicConfig(filename=f'log/detect_{datetime.now().strftime("%Y-%m-%d")}.log', 
                            filemode='a',
                            level=logging.INFO,
                            format='%(asctime)s - %(message)s')
        logging.info(f'Using device : {device}')
        
        detect = Detect(input_dir=input_dir, output_dir=output_dir)
        detect.run_detect()

        print("\nExecution completed successfully.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)