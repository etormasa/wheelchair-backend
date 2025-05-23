from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
import cv2
from ultralytics import YOLO
from screeninfo import get_monitors
import numpy as np
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolución de pantalla
monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

# Modelos
model_custom = YOLO("runs/detect/train/weights/best.pt")
model_coco = YOLO("yolov8n.pt")

# Colores
colors = {
    'wheelchairs': (0, 255, 0),
    'person': (255, 0, 0),
    'car': (0, 0, 255)
}

# Conteo total acumulado
conteo_total = {
    "peatones_dia": 0,
    "vehiculos": 0,
    "silla_ruedas": 0
}

# Conteo actual por frame
conteo_actual = {
    "person": 0,
    "car": 0,
    "wheelchairs": 0
}

# Track IDs únicos ya contados
ids_contados = {
    "person": set(),
    "car": set(),
    "wheelchairs": set()
}

def resize_with_aspect_ratio(image, target_width, target_height):
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    result = cv2.copyMakeBorder(
        resized,
        top=(target_height - new_h) // 2,
        bottom=(target_height - new_h + 1) // 2,
        left=(target_width - new_w) // 2,
        right=(target_width - new_w + 1) // 2,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )
    return result

def generate_frames():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        success, frame = cap.read()
        if not success:
            break

        for k in conteo_actual:
            conteo_actual[k] = 0

        # Modelo personalizado (wheelchairs)
        results_custom = model_custom.track(frame, persist=True)
        for r in results_custom:
            for box in r.boxes:
                track_id = int(box.id[0]) if box.id is not None else None
                if track_id is None:
                    continue

                label = "wheelchairs"
                if track_id not in ids_contados[label]:
                    ids_contados[label].add(track_id)
                    conteo_total["silla_ruedas"] += 1

                conteo_actual[label] += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                color = colors[label]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Modelo COCO (person, car)
        results_coco = model_coco.track(frame, persist=True)
        for r in results_coco:
            for box in r.boxes:
                track_id = int(box.id[0]) if box.id is not None else None
                if track_id is None:
                    continue

                cls_id = int(box.cls[0])
                label = model_coco.names[cls_id]
                if label not in ['person', 'car']:
                    continue

                if track_id not in ids_contados[label]:
                    ids_contados[label].add(track_id)
                    if label == "person":
                        conteo_total["peatones_dia"] += 1
                    elif label == "car":
                        conteo_total["vehiculos"] += 1

                conteo_actual[label] += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                color = colors[label]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frame = resize_with_aspect_ratio(frame, screen_width, screen_height)

        y = 40
        for label, total in conteo_actual.items():
            cv2.putText(frame, f"{label}: {total}", (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, colors[label], 2)
            y += 40

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
def index():
    return HTMLResponse(
        """
        <html>
        <head>
            <title>Detección en Tiempo Real</title>
        </head>
        <body style="margin:0; background-color:black;">
            <img src="/video_feed" style="width:100vw; height:100vh; object-fit:contain;">
        </body>
        </html>
        """
    )

@app.get("/api/stats")
def api_stats():
    ahora = datetime.now()
    return JSONResponse({
        "hora": ahora.strftime("%H:%M"),
        "fecha": ahora.strftime("%d/%m/%Y"),
        "peatones_cruzando": conteo_actual["person"],
        "peatones_dia": conteo_total["peatones_dia"],
        "vehiculos": conteo_total["vehiculos"],
        "silla_ruedas": conteo_total["silla_ruedas"]
    })
