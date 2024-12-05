from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os

# 이미지 크롭시 이미지 사이에 일정한 간격이 필요함 - 고정 측정 카메라가 있으면 좋을것으로 판단
def image_crop(image_path_folder):
	test_images =[]
	filename_list = []

	# 지원되는 이미지 파일 확장자
	supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp')

	# 폴더 내 모든 이미지 파일 읽기
	image_files = [f for f in os.listdir(image_path_folder) if f.lower().endswith(supported_extensions)]

	for filename in image_files:
		image_path = os.path.join(image_path_folder, filename)

		img = Image.open(image_path)
		image = np.array(img) # opencv 처리가 가능하도록 변환
		height, w, c = image.shape
		hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
		
		# 무채색 범위 정의 (채도 값이 낮은 경우)
		lower_gray = (0, 0, 50)    # H=0, S=0, V=50 (밝은 회색부터)
		upper_gray = (180, 50, 255) # H=180, S=50, V=255 (밝은 무채색 영역)
		
		mask_gray = cv2.inRange(hsv, lower_gray, upper_gray) # 무채색 마스크 생성
		result_gray = cv2.bitwise_and(image, image, mask=mask_gray) # 무채색 영역 추출
		blurred = cv2.GaussianBlur(result_gray, (5, 5), 0) # 블러링
		
		# ROI 설정 (예: 특정 사각형 영역)
		roi_x, roi_y, roi_w, roi_h = w//4, (height//3)+70, (height//2)*2, height//2 # 관심 영역 좌표와 크기
		roi = blurred[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
		
		edges = cv2.Canny(roi, 40, 140) # 엣지 검출
		contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 엣지로부터 컨투어(윤곽선) 찾기

		rectangles =[]
		
		for contour in contours:
			x,y,w,h = cv2.boundingRect(contour)
			#if w > 90 and h > 90: # 너비와 높이 조건을 추가할 경우 적절한 이미지 크기로 자름
			rectangles.append((x,y,w,h))
		
		if len(rectangles) > 1 :
			x,y,w,h = rectangles[0] # 특정 사각형 1개만 지정
			
			half_size = height//4

			center_x = int(roi_x + x + w // 2)
			center_y = int(roi_y + y + h // 2)
			
			x_start = max(center_x - half_size, 0)
			y_start = max(center_y - half_size, 0)
			x_end = x_start + (height//2)
			y_end = y_start + (height//2)

			cropped_image = image[y_start:y_end, x_start:x_end] # 원본 이미지에서 ROI 자르기
			pil_image = Image.fromarray(cropped_image)
			
			test_images.append(pil_image)
			filename_list.append(filename)

	return test_images, filename_list