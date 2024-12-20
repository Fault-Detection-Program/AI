import os
import json
import torch
import config
import logging
import sys
from PIL import ImageFile
from datetime import datetime
from dataset import get_transforms
from efficientnet_pytorch.model import EfficientNet
from preprocessor import auto_adjustments, opencv_processing
from utils.annotation import image_annotation
from utils.crop import image_crop
from config import PATH, MODEL
from collections import defaultdict

ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Detect:
    def __init__(self, base_dir, input_dir, output_dir):
        self.base_dir = base_dir
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.model_name = self.base_dir + MODEL['MODEL_NAME']
        self.weights_dir = self.base_dir + PATH['WEIGHTS_PATH']
        self.labels_file = self.base_dir + PATH['LABELS']
        self.weights_file = os.path.join(self.weights_dir, self.model_name + '.pth')
        self.tfms = get_transforms(EfficientNet.get_image_size(self.model_name))

    def run_detect(self):
        self.labels_map = json.load(open(self.labels_file))
        self.labels_map = [self.labels_map[str(i)] for i in range(len(self.labels_map))]

        if device.type == 'cpu':
            model = EfficientNet.from_pretrained(self.model_name, num_classes=len(self.labels_map))
            model.load_state_dict(torch.load(self.weights_file, map_location=device))
            
        else:
            model = EfficientNet.from_pretrained(self.model_name, 
                                                 weights_path=self.weights_file,
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

        # 이미지 파일 리스트 : ['MFR01328AE;2408221189A_R_OK.png', 'MFR01327AE;2409110702A_R_NG.png', ....]
        img_files = [f for f in os.listdir(self.input_dir) if f.endswith(('.jpg', '.png')) and f not in img_list]
        image_list, filename_list, coord_list = image_crop(self.input_dir) # 크롭된 PIL 이미지, 파일명 호출

        # 파일명 기준 라벨 그룹화
        label_map = defaultdict(list)
        for img, filename, coord in zip(image_list, filename_list, coord_list):
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
                label_map[original_filename].append({"coords": coord, "label": label})
        
            except Exception as e:
                logging.info(f'{filename} : {e}')

        image_annotation(img_files, label_map, self.input_dir, self.output_dir) # 이미지 label 표시

if __name__ == '__main__':
    try:
        base_dir = sys.argv[1]
        input_dir = sys.argv[2]
        output_dir = sys.argv[3]
            
        log_path = base_dir + PATH['Log']

        logging.basicConfig(filename= f'{log_path}/detect_{datetime.now().strftime("%Y-%m-%d")}.log',
                            filemode='a',
                            level=logging.INFO,
                            format='%(asctime)s - %(message)s')
        logging.info(f'Using device : {device}')

        detect = Detect(base_dir=base_dir, input_dir=input_dir, output_dir=output_dir)
        detect.run_detect()

        print("\n Defect FPCB Detection Completed.")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)