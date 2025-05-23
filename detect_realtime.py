from ultralytics import YOLO
import cv2
from screeninfo import get_monitors

# Detectar resolución del monitor principal
monitor = get_monitors()[0]
screen_width = monitor.width
screen_height = monitor.height

# Cargar modelos
model_custom = YOLO("runs/detect/train/weights/best.pt")  # Detecta 'wheelchairs'
model_coco = YOLO("yolov8n.pt")  # Detecta 'person', 'car', etc.

# Colores por clase
colors = {
    'wheelchairs': (0, 255, 0),  # verde
    'person': (255, 0, 0),       # azul
    'car': (0, 0, 255)           # rojo
}

counts = {
    'wheelchairs': 0,
    'person': 0,
    'car': 0
}

# Función para escalar manteniendo relación de aspecto
def resize_with_aspect_ratio(image, target_width, target_height):
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    # Rellenar bordes negros
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

# Abrir cámara
cap = cv2.VideoCapture(0)

# Crear ventana a pantalla completa
cv2.namedWindow("Detección combinada", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Detección combinada", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    for k in counts:
        counts[k] = 0

    # Detectar wheelchairs (modelo personalizado)
    results_custom = model_custom(frame, stream=True)
    for r in results_custom:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = "wheelchairs"
            color = colors[label]
            counts[label] += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Detectar person y car (modelo COCO)
    results_coco = model_coco(frame, stream=True)
    for r in results_coco:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model_coco.names[cls_id]
            if label in ['person', 'car']:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = colors[label]
                counts[label] += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Ajustar frame a pantalla completa (sin deformar)
    frame_fullscreen = resize_with_aspect_ratio(frame, screen_width, screen_height)

    # Mostrar conteo por clase
    y = 40
    for label, total in counts.items():
        cv2.putText(frame_fullscreen, f"{label}: {total}", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, colors[label], 2)
        y += 40

    cv2.imshow("Detección combinada", frame_fullscreen)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
