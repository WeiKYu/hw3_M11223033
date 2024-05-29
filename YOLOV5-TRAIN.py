import os
import sys
from pathlib import Path
import imgaug.augmenters as iaa
import torch
# 設定 PYTHONPATH
yolov5_path = 'C:/Users/ediso/Desktop/ML_project3/yolov5'
sys.path.append(yolov5_path)

from models.experimental import attempt_load
from utils.general import check_dataset, increment_path
from utils.plots import plot_results
from utils.callbacks import Callbacks
from train import run
from val import run as val_run  # 導入驗證函數

# 設置資料夾路徑
train_images_folder = 'C:/Users/ediso/Desktop/ML_project3/container/Train'
train_annotations_folder = 'C:/Users/ediso/Desktop/ML_project3/container/Train_xml'
val_images_folder = 'C:/Users/ediso/Desktop/ML_project3/container/Verify'
val_annotations_folder = 'C:/Users/ediso/Desktop/ML_project3/container/Verify_xml'
test_images_folder = 'C:/Users/ediso/Desktop/ML_project3/container/Test'
test_annotations_folder = 'C:/Users/ediso/Desktop/ML_project3/container/Test_xml'

# YOLOv5 資料集目錄
dataset_dir = 'C:/Users/ediso/Desktop/ML_project3/dataset'
output_dir = 'C:/Users/ediso/Desktop/ML_project3'
top_predictions_dir = os.path.join(output_dir, 'top_predictions')
best_model_path = os.path.join(output_dir, 'best.pt')

os.makedirs(os.path.join(dataset_dir, 'images/train'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'images/val'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'images/test'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'labels/train'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'labels/val'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'labels/test'), exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(top_predictions_dir, exist_ok=True)

# 生成 data.yaml 文件
def create_data_yaml():
    yaml_content = f"""
train: {dataset_dir}/images/train
val: {dataset_dir}/images/val
test: {dataset_dir}/images/test

nc: 1  # number of classes
names: ['container']  # class names
"""
    with open(os.path.join(dataset_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content)

create_data_yaml()

# 資料增強
augmenters = iaa.Sequential([
    iaa.Fliplr(0.5),  # 水平翻轉50%的圖片
    iaa.Crop(percent=(0, 0.1)),  # 隨機裁剪圖片
    iaa.LinearContrast((0.75, 1.5)),  # 調整對比度
    iaa.Multiply((0.8, 1.2)),  # 調整亮度
    iaa.Affine(
        rotate=(-20, 20),  # 隨機旋轉
        shear=(-10, 10)  # 隨機剪切
    ),
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # 加入高斯雜訊
    iaa.PiecewiseAffine(scale=(0.01, 0.05))  # 局部仿射變換
])

def augment_image(image):
    return augmenters.augment_image(image)

# 訓練YOLOv5
def train_yolov5():
    # 準備訓練參數
    data = os.path.join(dataset_dir, 'data.yaml')
    epochs = 200  
    batch_size = 32  
    img_size = 640
    weights = 'yolov5s.pt'
    project = output_dir
    name = 'exp'
    save_dir = increment_path(Path(project) / name, exist_ok=False)

    # 創建保存目錄
    os.makedirs(save_dir, exist_ok=True)

    # 設定 early stopping
    patience = 30

    # 執行訓練
    run(data=data, epochs=epochs, batch_size=batch_size, imgsz=img_size, weights=weights, project=project, name=name, patience=patience, save_dir=save_dir)

    # 複製最好的模型
    best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        os.rename(best_model_path, os.path.join(output_dir, 'best.pt'))

    # 執行驗證
    results = val_run(data=data, weights=best_model_path, imgsz=img_size, save_json=True)

    # 提取評估指標
    mAP = results['mAP_0.5']
    recall = results['recall']
    precision = results['precision']
    f1 = results['f1']

    # 打印評估指標
    print(f"mAP: {mAP}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"F1-score: {f1}")


# 執行訓練
if __name__ == "__main__":
    train_yolov5()
