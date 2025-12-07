import cv2
import mediapipe as mp
import os
import numpy as np

# -------------------------------
# 1. MEDIA PIPE INITIALIZATION
# -------------------------------
mp_face_detection = mp.solutions.face_detection

# -------------------------------
# 2. LBPH TRAINER (LOAD OR TRAIN)
# -------------------------------
recognizer = cv2.face.LBPHFaceRecognizer_create()

DATASET_DIR = "dataset_faces"

def train_lbph():
    images = []
    labels = []
    label_map = {}
    label_id = 0

    for person in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_path):
            continue

        label_map[label_id] = person

        for img_file in os.listdir(person_path):
            img = cv2.imread(os.path.join(person_path, img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            images.append(img)
            labels.append(label_id)

        label_id += 1

    if images:
        recognizer.train(images, np.array(labels))
        print("LBPH Training complete.")
    return label_map

label_map = train_lbph()

# -------------------------------
# 3. REAL-TIME RECOGNITION
# -------------------------------
cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(frame_rgb)

        if results.detections:
            for detection in results.detections:

                # Extract bounding box
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                w_box = int(bbox.width * w)
                h_box = int(bbox.height * h)

                # Crop face safely
                x2 = max(0, x)
                y2 = max(0, y)
                face_crop = frame[y2:y2+h_box, x2:x2+w_box]

                if face_crop.size != 0:
                    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

                    # LBPH recognition
                    label, confidence = recognizer.predict(gray)

                    name = label_map.get(label, "Unknown")

                    # Draw results
                    cv2.rectangle(frame, (x2, y2), (x2+w_box, y2+h_box), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} ({confidence:.0f})",
                                (x2, y2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)

        cv2.imshow("MediaPipe + LBPH Face Recognition", frame)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
