import numpy as np
import time
import cv2
import os
import subprocess
from gtts import gTTS
from pydub import AudioSegment
AudioSegment.converter = "ffmpeg.exe"

# YOLO modelinin eğitildiği COCO sınıfı etiketleri yüklüyorum ve COCO veri seti üzerinde eğitilmiş YOLO nesne algılama modülünü tanımlıyorum.
LABELS = open("coco.names").read().strip().split("\n")

print("- Bilgilendirme - YOLO disk üzerinden yükleniyor...")
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

# YOLO'dan nesneleri görüntüden ayırt ederken ihtiyacımız olan *çıktı* katman adlarını belirledim.
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Ardından taramayı başlattım ve frameleri kaydetmek için bir list oluşturdum.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
frame_count = 0
start = time.time()
first = True
frames = []

# While döngüsü içerisinde kamerayı algılama ve nesneleri algılama işlemini hazırladım.
while True:
    frame_count += 1
    # Kareden kareye yakaladım, okudum ve bunları uygulama içinde gösterdim.
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frames.append(frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break
    if ret:
        key = cv2.waitKey(1)
        cv2.imshow("Webcam Input Scan", frame)
        if frame_count % 60 == 0:
            end = time.time()
            # Çerçeve boyutlarını yakaladım ve onu bir blob'a yani parçaya dönüştürdüm.
            (H, W) = frame.shape[:2]
            # Giriş görüntüsünden bir blob oluşturdum ve ardından
            # YOLO nesne dedektörünün ilerlemesini gerçekleştirerek
            # sınırlayıcı kutuları ve ilişkili olasılıkları verdim.
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                         swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)

            # Sırasıyla tespit edilen sınırlayıcı kutuları, doğrulukları ve sınıf kimlikleri listelerimizi tanımladım.
            boxes = []
            confidences = []
            classIDs = []
            centers = []

            # Katman çıktılarının her biri üzerinde döngü başlattım
            for output in layerOutputs:
                # Daha sonra algılamaların her birinin üzerinde döngü başlattım
                for detection in output:
                    # Mevcut algılanan nesnenin sınıf kimliğini ve güvenirliğini (yani olasılığını) çıkarttım.
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    # Tespit edilen olasılığın minimum olasılıktan daha büyük olması koşulu ile zayıf tahminleri filtreledim.
                    if confidence > 0.5:
                        # Sınırlayıcı kutunun koordinatlarını görüntünün boyutuna göre geri ölçekledim.
                        # YOLO aslında burada sınırlayıcı kutunun merkez (x, y) koordinatlarını
                        # ardından kutuların genişlik ve yüksekliğini döndürttüyor.
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        # Sınırlayıcı kutunun üst ve sol köşelerini elde etmek için
                        # merkez (x, y) koordinatlarını kullandım.
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        # Sınırlayıcı kutu koordinatları, doğruluk (olasılık) ve sınıf kimlikleri listelerini güncelledim.
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
                        centers.append((centerX, centerY))

            # Üst üste binen sınırlayıcı kutuları bastırmak için
            # maksimum olmayan bastırma yani Non-maxima suppression uyguladım.
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

            texts = []

            # En az bir nesne algılaması mevcut olduğundan emin oldum, pozisyonları aldım ve
            # sınırlandırılmış algılamalar dizini üzerinde döngü sağladım.
            # Sonrasında nesnelerin merkez noktasını değiştirerek kameranın yükseklik genişliğine göre konumlarını, pozisyonları ayarladım.
            if len(idxs) > 0:
                
                for i in idxs.flatten():
                    
                    centerX, centerY = centers[i][0], centers[i][1]

                    if centerX <= W/3:
                        W_pos = "sol "
                    elif centerX <= (W/3 * 2):
                        W_pos = "merkez "
                    else:
                        W_pos = "sağ "

                    if centerY <= H/3:
                        H_pos = "üst "
                    elif centerY <= (H/3 * 2):
                        H_pos = "orta "
                    else:
                        H_pos = "alt "

                    texts.append(H_pos + W_pos + LABELS[classIDs[i]])

            print(texts)

            if texts:
                description = ', '.join(texts)
                tts = gTTS(description, lang='tr')
                tts.save('tts.mp3')
                tts = AudioSegment.from_mp3("tts.mp3")
                subprocess.call(["ffplay", "-nodisp", "-autoexit", "tts.mp3"])


cap.release()
cv2.destroyAllWindows()
os.remove("tts.mp3")
