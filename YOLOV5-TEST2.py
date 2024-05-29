import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
import pytesseract
import os

# 設置Tesseract安裝路徑
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 加載已訓練好的YOLOv5模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/ediso/Desktop/ML_project3/exp6/weights/best.pt')

# 設置測試集圖片路徑和結果保存路徑
test_image_folder = "C:/Users/ediso/Desktop/ML_project3/container/Test"
results_folder = "C:/Users/ediso/Desktop/ML_project3/OCR-results"
os.makedirs(results_folder, exist_ok=True)

# 定義正確的貨櫃號碼，根據檔案名稱
correct_numbers_folder = "C:/Users/ediso/Desktop/pictest"
correct_numbers = {file_name: file_name.split('.')[0] for file_name in os.listdir(correct_numbers_folder)}

# 初始化計數器
correct_count = 0
total_count = 0

def preprocess_image(img):
    # 增加對比度
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)
    
    # 轉換為灰度圖像
    img = ImageOps.grayscale(img)
    
    # 二值化處理
    img = img.point(lambda x: 0 if x < 128 else 255, '1')
    
    return img

# 遍歷測試集的所有圖片
for image_file in os.listdir(test_image_folder):
    if image_file.endswith(".jpg") or image_file.endswith(".png"):
        image_path = os.path.join(test_image_folder, image_file)
        image = Image.open(image_path)
        
        # 使用YOLOv5模型進行物件偵測
        results = model(image_path)
        detected_objects = results.xyxy[0].cpu().numpy()  # 獲取偵測結果並轉換為NumPy陣列
        
        # 繪製檢測結果
        draw = ImageDraw.Draw(image)
        for obj in detected_objects:
            x1, y1, x2, y2, conf, cls = obj
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        
        # 保存視覺化結果
        image.save(os.path.join(results_folder, f"{image_file}_detected.jpg"))
        
        # 遍歷偵測到的物件
        for obj in detected_objects:
            x1, y1, x2, y2, conf, cls = obj
            cropped_img = image.crop((x1, y1, x2, y2))
            
            # 預處理裁剪出的圖片
            cropped_img = preprocess_image(cropped_img)
            
            # 使用Tesseract進行文字辨識
            recognized_text = pytesseract.image_to_string(cropped_img)
            recognized_text = ''.join(filter(str.isalnum, recognized_text))
            
            # 比較前11位數字
            correct_number = correct_numbers.get(image_file, "")
            if recognized_text[:11] == correct_number[:11]:
                correct_count += 1
            total_count += 1

# 計算準確率
accuracy = correct_count / total_count if total_count > 0 else 0
print(f"OCR 準確率: {accuracy:.2f}")