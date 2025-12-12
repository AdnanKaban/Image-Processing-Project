import torch
from matplotlib.pyplot import annotate
from ultralytics import YOLO
import cv2
import time

#Yolo modelinin yüklenmesi
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8l.pt").to(device)
#video giriş yolu
video_path="SampleVideo_LowQuality.mp4"
cap=cv2.VideoCapture(video_path)
#Cikti videosunu görüntüleme
width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps=cap.get(cv2.CAP_PROP_FPS)
out=cv2.VideoWriter("output.avi",cv2.VideoWriter_fourcc(*"XVID"),fps,(width,height))
prev_time = 0
seen_ids = set()

while cap.isOpened():
    success,frame=cap.read()
    if not success:
        break
#Yolo İle Takip
    results=model.track(frame,#giris goruntusu
                        persist=True,#takip idsi korunumu
                        conf=0.3,     #guven skoru minimum seviye
                        iou=0.5,#nesne kutularının ne kadar ortusmesi gerektigi
                        tracker="bytetrack.yaml",#takip algoritmasi konfigurasyonu

                        )
    #kutuşarı ve idleri ekran üzerine yazdır
    annotated_frame=results[0].plot()
    # FPS hesabi

    current_time = time.time()
    fps_live = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps_live)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Toplam arac sayisi (ID bazli)
    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy()
        for id in ids:
            seen_ids.add(int(id))

    cv2.putText(
        annotated_frame,
        f"Total Vehicles: {len(seen_ids)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )




    #goster ve kaydet
    cv2.imshow("YOLO v8 Tracking",annotated_frame)
    out.write(annotated_frame)
    if cv2.waitKey(1)& 0xFF ==ord("q"): #Q harfine basılınca cıkar
        break

cap.release()
out.release()
cv2.destroyAllWindows()
