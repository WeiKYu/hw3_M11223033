import torch
import cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageOps
import pytesseract
import os
import numpy as np
from collections import defaultdict
import re

# 設置Tesseract安裝路徑
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 加載已訓練好的YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/ediso/Desktop/ML_project3/exp6/weights/best.pt')

# 設置影片資料夾和結果保存路徑
video_folder = "C:/Users/ediso/Desktop/AVI"
results_folder = "C:/Users/ediso/Desktop/ML_project3/OCR-video-results"
frames_folder = "C:/Users/ediso/Desktop/ML_project3/OCR-frames"
os.makedirs(results_folder, exist_ok=True)
os.makedirs(frames_folder, exist_ok=True)

# 定義正確的貨櫃號碼，根據檔案名稱
correct_numbers_folder = "C:/Users/ediso/Desktop/pictest"
correct_numbers = {Path(file_name).stem: Path(file_name).stem for file_name in os.listdir(correct_numbers_folder)}

# 調試信息：列出正確的貨櫃號碼
print("Correct numbers dictionary:")
for k, v in correct_numbers.items():
    print(f"{k}: {v}")

# 設置字體大小
font_path = "arial.ttf"  # 確認系統中有此字體，或者選擇其他字體
font_size = 30  # 可以根據需要調整字體大小
font = ImageFont.truetype(font_path, font_size)

def is_valid_container_number(number):
    if len(number) < 11:
        return False
    if not number[10].isdigit():
        return False
    return calculate_check_digit(number[:10]) == int(number[10])

def calculate_check_digit(code):
    values = {'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17, 'H': 18, 'I': 19, 'J': 20,
              'K': 21, 'L': 23, 'M': 24, 'N': 25, 'O': 26, 'P': 27, 'Q': 28, 'R': 29, 'S': 30, 'T': 31,
              'U': 32, 'V': 34, 'W': 35, 'X': 36, 'Y': 37, 'Z': 38}
    try:
        s = sum(values[code[i]] * (2 ** i) for i in range(4)) + sum(int(code[i + 4]) * (2 ** (i + 4)) for i in range(6))
        return s % 11 % 10
    except (KeyError, ValueError):
        return -1  # 如果有無效字符或转换失败，返回-1

def adjust_box(x1, y1, x2, y2, scale=1.2):
    """
    调整框的大小。
    
    :param x1: 左上角x坐标
    :param y1: 左上角y坐标
    :param x2: 右下角x坐标
    :param y2: 右下角y坐标
    :param scale: 调整比例（默认为1.2）
    :return: 调整后的坐标
    """
    width = x2 - x1
    height = y2 - y1
    new_width = width * scale
    new_height = height * scale
    
    # 计算新坐标，使框保持中心
    new_x1 = x1 - (new_width - width) / 2
    new_y1 = y1 - (new_height - height) / 2
    new_x2 = x2 + (new_width - width) / 2
    new_y2 = y2 + (new_height - height) / 2
    
    return new_x1, new_y1, new_x2, new_y2

def preprocess_image_and_save(img, save_path):
    # 增加對比度
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)

    # 轉換為灰度圖像
    img = ImageOps.grayscale(img)

    # 自適應閾值二值化處理
    img = img.point(lambda x: 0 if x < 128 else 255, '1')

    # 保存二值化處理後的圖片供檢查
    img.save(save_path)

    return img

results_log = os.path.join(results_folder, "results_log2.txt")
with open(results_log, "w", encoding="utf-8") as log_file:
    overall_correct_count = 0
    overall_total_count = 0
    
    for video_file in os.listdir(video_folder):
        if video_file.endswith(".mp4") or video_file.endswith(".avi"):
            video_path = os.path.join(video_folder, video_file)
            video_capture = cv2.VideoCapture(video_path)

            frame_width = int(video_capture.get(3))
            frame_height = int(video_capture.get(4))
            output_path = os.path.join(results_folder, f"{Path(video_file).stem}_processed.avi")
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width, frame_height))

            frame_results = defaultdict(int)
            correct_count = 0
            total_count = 0
            frame_idx = 0

            # 创建子资料夹来保存该影片的图片
            video_stem = Path(video_file).stem
            video_frames_folder = os.path.join(frames_folder, video_stem)
            os.makedirs(video_frames_folder, exist_ok=True)

            # 从正确的货柜号码文件中提取对应的正确号码
            correct_number = correct_numbers.get(video_stem, "")

            # 调试信息：确认从文件名中提取的正确号码
            print(f"Video: {video_file}, Extracted stem: {video_stem}, Correct number: {correct_number}")

            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    break

                # 将图片转换为PIL格式
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                # 使用YOLOv5模型进行物件检测
                results = model(image)
                detected_objects = results.xyxy[0].cpu().numpy()
                
                draw = ImageDraw.Draw(image)
                for obj in detected_objects:
                    x1, y1, x2, y2, conf, cls = obj
                    # 调整检测框大小
                    x1, y1, x2, y2 = adjust_box(x1, y1, x2, y2, scale=1.2)
                    # 绘制检测框
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                    cropped_img = image.crop((x1, y1, x2, y2))
                    # 保存裁剪后的图片供调试
                    cropped_output_path = os.path.join(video_frames_folder, f"{video_stem}_frame_{frame_idx}_cropped.jpg")
                    cropped_img.save(cropped_output_path)
                    
                    # 设置 Tesseract 参数
                    custom_config = r'--oem 3 --psm 6'
                    processed_img = preprocess_image_and_save(cropped_img, os.path.join(video_frames_folder, f"{video_stem}_frame_{frame_idx}_processed.jpg"))
                    recognized_text = pytesseract.image_to_string(processed_img, config=custom_config)
                    recognized_text = ''.join(filter(str.isalnum, recognized_text))
                    
                    if is_valid_container_number(recognized_text):
                        frame_results[recognized_text[:11]] += 1
                        draw.text((x1, y1 - 10), recognized_text[:11], fill="red", font=font)
                        
                        # 计算总数和正确数
                        total_count += 1
                        if recognized_text[:11] == correct_number[:11]:
                            correct_count += 1

                # 保存每帧图片
                frame_output_path = os.path.join(video_frames_folder, f"{video_stem}_frame_{frame_idx}.jpg")
                image.save(frame_output_path)
                frame_idx += 1
                
                processed_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                out.write(processed_frame)

            video_capture.release()
            out.release()

            # 进行多数决，决定最终的货柜号码
            if frame_results:
                final_number = max(frame_results, key=frame_results.get)
                log_file.write(f"Video: {video_file}, Final Container Number: {final_number}\n")
                
                if final_number == correct_number:
                    log_file.write(f"Video: {video_file}, Recognized correctly: {final_number}\n")
                    overall_correct_count += 1
                else:
                    log_file.write(f"Video: {video_file}, Recognized incorrectly: {final_number} (correct: {correct_number})\n")
            else:
                log_file.write(f"Video: {video_file}, No valid container number detected\n")
            
            overall_total_count += 1

            # 计算影片中的OCR准确率
            accuracy = correct_count / total_count if total_count > 0 else 0
            log_file.write(f"Video: {video_file}, OCR Accuracy: {accuracy:.2f}\n")

    # 计算并记录整体准确率
    overall_accuracy = overall_correct_count / overall_total_count if overall_total_count > 0 else 0
    log_file.write(f"Overall OCR Accuracy: {overall_accuracy:.2f}\n")

print("所有结果已记录在 results_log.txt 中。")
print(f"Overall OCR Accuracy: {overall_accuracy:.2f}")