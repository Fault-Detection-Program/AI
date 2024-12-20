import os
import json
import torch
import logging
import sys
from PIL import ImageFile
from datetime import datetime
from efficientnet_pytorch.model import EfficientNet
from utils.preprocessor import auto_adjustments, opencv_processing
from utils.annotation import image_annotation
from utils.crop import image_crop
from dataset import get_transforms
from config import PATH, MODEL
from collections import defaultdict

ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Detect:
    def __init__(self, base_dir, input_dir, output_dir):
        self.base_dir = base_dir
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.model_name = MODEL['MODEL_NAME']
        self.weights_dir = self.base_dir + PATH['WEIGHTS_PATH']
        self.labels_file = self.base_dir + PATH['LABELS']
        self.weights_file = os.path.join(self.weights_dir, self.model_name + '.pth')
        self.image_size = EfficientNet.get_image_size(self.model_name)
        self.tfms = get_transforms(self.image_size)

    # EfficientNet 모델 로드
    def load_model(self):
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

        return model
    
    # output 경로에 존재하지 않는 이미지 리스트 반환 및 폴더 생성
    def image_list_check(self):
        img_list = set()
        
        for sub_dir in ['OK', 'PASS', 'NG']:
            sub_path = os.path.join(self.output_dir, sub_dir)
            os.makedirs(sub_path, exist_ok=True)
            if os.path.exists(sub_path):
                img_list.update(os.listdir(sub_path))

        return img_list

    # 이미지 전처리
    def image_preprocess(self, img):
        img = auto_adjustments(img)  # 자동 밝기, 대비, 선명도 조정
        img = opencv_processing(img)  # 노이즈 제거
        img = img.convert('L')  # Grayscale 변환
        img = img.convert('RGB')  # 3차원 변환
        img = self.tfms(img).unsqueeze(0)

        return img

    # 이미지 불량 검출 실행
    def run_detect(self):
        model = self.load_model()
        img_list = self.image_list_check()

        img_files = [f for f in os.listdir(self.input_dir) if f.endswith(('.jpg', '.png')) and f not in img_list]
        
        # 이미지 크롭(이미지 리스트, 파일명 리스트, 좌표 리스트 반환) 및 전처리
        images, filenames, coords = image_crop(self.input_dir) 
        images = [self.image_preprocess(img).to(device) for img in images]

        label_map = defaultdict(list)

        for img, filename, coord in zip(images, filenames, coords):
            try:
                with torch.no_grad():
                    logits = model(img)
                    pred = torch.argmax(logits, dim=1).item()
                    prob = torch.softmax(logits, dim=1)[0, pred].item()
                    label = self.labels_map[pred]

                logging.info('-' * 100)
                logging.info(f'Image File : {filename}')
                logging.info('{:<75} ({:.2f}%)'.format(label, prob*100))
                logging.info('-' * 100)

                original_filename = filename.rsplit('_', 1)[0]
                label_map[original_filename].append({"coords": coord, "label": label})

            except Exception as e:
                logging.info(f'{filename} : {e}')

        image_annotation(img_files, label_map, self.input_dir, self.output_dir)

if __name__ == '__main__':
    try:
        base_dir = sys.argv[1]
        input_dir = sys.argv[2]
        output_dir = sys.argv[3]
        log_path = base_dir + PATH['LOG']

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