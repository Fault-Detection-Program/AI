from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from config import PATH #경로
import numpy as np
import cv2
import os

# 이미지를 주로 부품이 위치한 자리를 기준으로 crop 시켰을 때
def crop_and_save(image, crop_coords, filename_prefix, index):
    y, x, h, w = crop_coords
    cropped_image = image[y:y+h, x:x+w]
    pil_image = Image.fromarray(cropped_image)
    output_filename = f"{filename_prefix}_{index}"
    return pil_image, output_filename

# 이미지 크롭시 이미지 사이에 일정한 간격이 필요함 - 고정 측정 카메라가 있으면 좋을것으로 판단
def image_crop(image_path_folder):
	test_images =[]
	filename_list = []
	crop_coord_listall = []

	# 지원되는 이미지 파일 확장자
	supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp')

	# 폴더 내 모든 이미지 파일 읽기
	image_files = [f for f in os.listdir(image_path_folder) if f.lower().endswith(supported_extensions)]

	for filename in image_files:
		image_path = os.path.join(image_path_folder, filename)

		# 이미지 열기
		img = Image.open(image_path)

		image = np.array(img) # opencv 처리가 가능하도록 변환
		
		height, weight, c = image.shape

		x,y,w,h = weight//5 - 30, height//5 - 40, (weight//2)-(weight//5), height-(height//5*2) #왼쪽 부품 대략적인 위치
		x2,y2,w2,h2 = (weight//7*3)+(weight//10), height//5 - 50, (weight//2)-(weight//5), height-(height//5*2) #오른쪽 부품 대략적인 위치

		# 좌표와 작업 한번에 처리
		crop_coords_list = [(y, x, h, w), (y2, x2, h2, w2)]
		for i, crop_coords in enumerate(crop_coords_list):
			filename2 = os.path.splitext(filename)[0] #filename.replace(".png", "")
			pil_image, output_filename = crop_and_save(image, crop_coords, filename2, i)
			test_images.append(pil_image)
			filename_list.append(output_filename)
			crop_coord_listall.append(crop_coords)

	return test_images, filename_list, crop_coord_listall
