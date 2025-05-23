from ultralytics import YOLO

# Carga modelo base (puedes usar yolov8s.pt si tu GPU es potente)
model = YOLO("yolov8n.pt")

# Entrena el modelo con tu archivo data.yaml
model.train(data="data.yaml", epochs=50, imgsz=640)
         