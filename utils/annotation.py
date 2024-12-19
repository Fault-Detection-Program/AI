import os
from PIL import Image, ImageDraw, ImageFont

# 우선순위에 따라 최종 라벨 결정 및 파일 복사
def image_annotation(img_files, label_map, input_dir, output_dir):
    ok_dir = os.path.join(output_dir, 'OK')
    pass_dir = os.path.join(output_dir, 'PASS')
    ng_dir = os.path.join(output_dir, 'NG')
    
    label_priority = {'NG': 1, 'PASS': 2, 'OK': 3} # label 우선순위
    
    font_path = "C:/Windows/Fonts/arial.ttf"  # TTF 폰트 파일 경로
    font_size = 100
    font = ImageFont.truetype(font_path, font_size)
    
    # 원본 파일 경로 찾기
    for filename, labels in label_map.items():
        img_file = next((file for file in img_files if filename in file), None)
        if img_file:
            img_path = os.path.join(input_dir, img_file)

        with Image.open(img_path) as img:
            draw = ImageDraw.Draw(img)
        
            # 좌표를 하나씩 그리기
            for label_entry in labels:
                coords = label_entry['coords']  # (y, x, h, w)
                label = label_entry['label']
                
                if label in ['NG', 'PASS']:
                    y, x, h, w = coords # 좌표 변환
                    
                    # 박스 그리기
                    box_color = "red" if label == 'NG' else "green"
                    draw.rectangle((x, y, x+w, y+h), outline=box_color, width=15)
                    draw.text((x+70, y+50), label, fill=box_color, font=font)

            # 최종 라벨 결정
            final_label_entry = min(labels, key=lambda x: label_priority[x['label']])
            final_label = final_label_entry['label']

            # 복사 대상 디렉토리 선택
            if final_label == 'OK': 
                target_dir = ok_dir 
            elif final_label == 'PASS':
                target_dir = pass_dir
            else:
                target_dir = ng_dir

            # 저장 경로
            target_path = os.path.join(target_dir, img_file)

            # 표시된 이미지 저장
            img.save(target_path)