import os
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 선명도 판단 함수 (Laplacian Variance)
def calculate_sharpness(img):
    # 이미지 그레이스케일로 변환
    img_gray = np.array(img.convert('L'))
    # 라플라시안(두 번째 미분) 계산
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    sharpness = laplacian.var()  # 분산 값으로 선명도 평가
    return sharpness

# 대비 판단 함수 (히스토그램 기반)
def calculate_contrast(img):
    img_gray = np.array(img.convert('L'))
    # 히스토그램의 최대값과 최소값을 사용하여 대비 계산
    contrast = np.std(img_gray)  # 표준편차를 대비의 지표로 사용
    return contrast

# 밝기 판단 함수 (평균 밝기)
def calculate_brightness(img):
    img_gray = np.array(img.convert('L'))
    brightness = np.mean(img_gray)  # 평균 밝기로 판단
    return brightness

# 밝기 보호 (이미지에서 너무 밝은 부분을 유지하고 나머지 밝기 조정)
def protect_bright_areas(img, brightness_factor=1.5, max_brightness_threshold=220):
    # 이미지를 numpy 배열로 변환
    img_np = np.array(img)
    
    # 밝은 부분은 보호하고, 나머지는 밝기 조정
    img_np = np.clip(img_np * brightness_factor, 0, 255)  # 밝기 조정 (밝은 부분을 과도하게 밝히지 않도록)

    # 너무 밝은 부분은 원본 이미지로 복원
    bright_areas = img_np > max_brightness_threshold
    img_np[bright_areas] = np.array(img)[bright_areas]  # 밝은 부분은 원본으로 유지

    # 변경된 이미지를 다시 PIL 이미지로 변환
    img_result = Image.fromarray(np.uint8(img_np))
    return img_result

# 밝기, 대비, 선명도를 자동으로 조정하는 함수
def auto_adjustments(img):
    sharpness = calculate_sharpness(img)
    contrast = calculate_contrast(img)
    brightness = calculate_brightness(img)

    # 선명도 자동 조정 (분산이 낮으면 선명도를 높임)
    if sharpness < 100:  # 선명도가 낮으면 선명도 증가
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(2.0)  # 선명도 2배 증가

    # 대비 자동 조정 (대비가 낮으면 대비 증가)
    if contrast < 50:  # 대비가 낮으면 대비 증가
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)  # 대비 1.5배 증가

    # 밝기 자동 조정 (어두운 이미지의 밝기 증가)
    if brightness < 100:  # 밝기가 낮으면 밝기 증가
        img = protect_bright_areas(img, brightness_factor=1.5)  # 밝기 증가 (밝은 영역 보호)
    
    return img

def opencv_processing(img):
    # 이미지를 OpenCV로 변환
    img_np = np.array(img)
    if len(img_np.shape) == 2:  # 이미 그레이스케일인 경우 RGB로 변환
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    # 노이즈 제거 (GaussianBlur)
    img = cv2.GaussianBlur(img_np, (5, 5), 0) #5,5 #sigmax=0 -> 블러강도 세밀한 조정

    return Image.fromarray(img)#gamma_corrected_img

def image_pre(folder_path):
    # 폴더 내의 모든 파일을 순차적으로 처리
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # 원하는 이미지 포맷으로 필터링
            image_path = os.path.join(folder_path, filename)
            
            # 이미지 열기
            img = Image.open(image_path)
            
            # 자동 밝기, 대비, 선명도 조정
            img_processed = auto_adjustments(img)
            
            # OpenCV 기반 추가 처리
            img_opencv_adjusted = opencv_processing(img_processed)
            
            # 이미지를 그레이스케일로 변환 (여기서 'L' 모드로 변환)
            img_grayscale = img_opencv_adjusted.convert('L') 
            
            return img_grayscale

def img_train_pre(input_dir, output_dir, folder):
    # 입력 디렉토리의 모든 하위 폴더 및 파일 탐색
    input_dir = input_dir+'/'+folder
    base_folder_name = os.path.basename(input_dir)
    output_dir = output_dir+'/'+'train'

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.jpg', '.png')):  # 이미지 파일 필터링
                folder_path = root.replace('\\', '/')
                input_path = os.path.join(root, file)
                
                # 출력 디렉토리 경로 구성
                relative_path = os.path.relpath(input_path, input_dir)
                
                output_path = os.path.join(output_dir, relative_path)
                
                output_dir_path = os.path.dirname(output_path)
                if not os.path.exists(output_dir_path):
                    os.makedirs(output_dir_path)  # 경로가 없으면 생성
                
                # 이미지 전처리
                # 이미지 열기
                img = Image.open(input_path)
                
                # 자동 밝기, 대비, 선명도 조정
                img_processed = auto_adjustments(img)
                
                # OpenCV 기반 추가 처리 -> 이미지 붉어지는 현상 존재
                img_opencv_adjusted = opencv_processing(img_processed)
                
                # 이미지를 그레이스케일로 변환 (여기서 'L' 모드로 변환)
                img_grayscale = img_opencv_adjusted.convert('L')
                
                # 전처리된 이미지를 저장 (PNG로 저장)
                img_grayscale.save(output_path)

# 이미지 전처리 후 grayscale 변환이 가장 높은 성능을 보임.