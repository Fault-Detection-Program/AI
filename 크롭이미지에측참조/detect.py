# -*- coding: utf-8 -*-

import os
import json
import torch
import config
import logging
import traceback
from PIL import Image
from shutil import copy2
from datetime import datetime
from dataset import get_transforms
from efficientnet_pytorch.model import EfficientNet
from preprocessor.image_preprocess import auto_adjustments, opencv_processing
from config import PATH, MODEL
from crop_image import image_crop
from collections import defaultdict

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

        state_dict = torch.load(self.model_weights_file, map_location=device)
        # 학습 모델 호출
        model = EfficientNet.from_pretrained(self.model_name, 
                                             #weights_path=self.model_weights_file, 
                                             num_classes=len(self.labels_map))

        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()

        # output 경로에 존재하지 않는 이미지 리스트만 img_files 변수에 할당
        img_list = set()
        for sub_dir in ['OK', 'PASS', 'NG']:
            sub_path = os.path.join(self.output_dir, sub_dir)
            os.makedirs(sub_path, exist_ok=True)

            if os.path.exists(sub_path):
                img_list.update(os.listdir(sub_path))

        # 이미지 파일 리스트 : ['MFR01328AE;2408221189A_R_OK.png', 'MFR01327AE;2409110702A_R_NG.png', ....]
        img_files = [f for f in os.listdir(self.input_dir) if f.endswith(('.jpg', '.png')) and f not in img_list]
        imagelist, filenamelist = image_crop(self.input_dir) # 크롭된 PIL 이미지, 파일명 호출
        # img_files = [f for f in filenamelist if f not in img_list] 

        # 라벨 우선순위 정의
        label_priority = {'NG': 1, 'PASS': 2, 'OK': 3}

        # 파일명 기준 라벨 그룹화
        label_map = defaultdict(list)
        for img, filename in zip(imagelist, filenamelist):
            try:
                # 각 바코드 이미지당 2개의 이미지 모두 예측 후 우선순위에 따른 예측 판단 진행되도록 함
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
                logging.info(f'Image File : {filename}')
                logging.info('{:<75} ({:.2f}%)'.format(label, prob*100))
                logging.info('-' * 100)

                # 그룹화: 파일명 기준 라벨 추가
                original_filename = filename.rsplit('_', 1)[0]  # 'data_file_1.png' -> 'data_file'
                label_map[original_filename].append(label)

            except Exception as e:
                logging.info(f'{img_file} : {e}')

        ok_dir = os.path.join(self.output_dir, 'OK')
        pass_dir = os.path.join(self.output_dir, 'PASS')
        ng_dir = os.path.join(self.output_dir, 'NG')

        # 우선순위에 따라 최종 라벨 결정 및 파일 복사
        for filename, labels in label_map.items():
            
            # 최종 라벨 결정
            final_label = min(labels, key=lambda x: label_priority[x])
            
            # 복사 대상 디렉토리 선택
            if final_label == 'OK':
                target_dir = ok_dir
            elif final_label == 'PASS':
                target_dir = pass_dir
            else:  # final_label == 'NG'
                target_dir = ng_dir

            # 원본 파일 경로
            img_file = next((file for file in img_files if filename in file), None)
            if img_file:
                img_path = os.path.join(input_dir, img_file)
                copy2(img_path, target_dir)
            else:
                return

if __name__ == '__main__':
    import sys
    try:
        path = sys.argv[1]
        config.set_config(path)
        input_dir = sys.argv[2]#PATH['INPUT_PATH']
        output_dir = sys.argv[3]#PATH['OUTPUT_PATH']
    
        log_path = PATH['Log']

        logging.basicConfig(filename= f'{log_path}/detect_{datetime.now().strftime("%Y-%m-%d")}.log', 
                            filemode='a',
                            level=logging.INFO,
                            format='%(asctime)s - %(message)s')
    
        detect = Detect(input_dir=input_dir, output_dir=output_dir)
        detect.run_detect()

        print("\nExecution completed successfully.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)