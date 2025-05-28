from tensorflow.keras.models import load_model
import cv2
import numpy as np
from tensorflow.keras.models import load_model


model = load_model("gesture_model.h5")


class_names = ['01_palm', '02_fist', '03_thumb', '04_index', '05_ok', 
               '06_peace', '07_stop', '08_point', '09_hold', '10_cup']

# camera start
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera açılamadı")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kare alınamadı")
        break

    
    x_start, y_start = 100, 100
    x_end, y_end = 400, 400
    roi = frame[y_start:y_end, x_start:x_end]

   
    img = cv2.resize(roi, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # (1,64,64,3)

  
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    
    text = f"{class_names[class_index]} ({confidence*100:.2f}%)"
    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)

    
    cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)

  
    cv2.imshow("Gerçek Zamanlı El Hareketi", frame)

 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

