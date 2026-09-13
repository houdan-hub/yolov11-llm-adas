#%%
from ultralytics import YOLO  as YOLO
import warnings
warnings.filterwarnings('ignore')
#%%
# 数据集的信息

data_yaml_path = r'./TS/data.yaml'
#%%
if __name__ == '__main__':
    model = YOLO('yolo11m.yaml')
    #训练模型
    results = model.train(data=data_yaml_path, # yaml文件的地址
                          imgsz=640,
                          epochs=150,
                          batch=4,
                          workers=16,
                          amp=True, 
                          project='runs/V11train',
                          name='exp',
                          cache="ram",
                          )
#%%
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('runs/V11train/exp/weights/best.pt')

# 对单张图片进行检测
results = model('')  

# 可视化检测结果
results[0].show()
# 或者保存检测结果
results[0].save(save_dir='detect_results/')  # 保存到detect_results目录下


