from ultralytics import YOLO
import os



os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Load your custom YOLO26 architecture
model = YOLO(r"F:\20000\ultralytics-main\ultralytics\cfg\models\26\yolo26-wafer.yaml")


# Train with custom losses
train_results = model.train(
    data=r"F:\200tehapamu\dataset\jingyuan1\jingyuan\data.yaml",
    epochs=100,
    workers=0,
    batch=4,
    imgsz=320,
    device="cuda",
    # amp=True,
)

print("Training completed with custom losses!")