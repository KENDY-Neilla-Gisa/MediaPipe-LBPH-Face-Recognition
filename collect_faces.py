import cv2
import os
import mediapipe as mp

name = input("Enter person name: ")

SAVE_DIR = f"dataset_faces/{name}"
os.makedirs(SAVE_DIR, exist_ok=True)

mp_face = mp.solutions.face_detection

cap = cv2.VideoCapture(0)
count = 0

with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                h, w, _ = frame.shape

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                w_box = int(bbox.width * w)
                h_box = int(bbox.height * h)

                face = frame[y:y+h_box, x:x+w_box]

                if face.size > 0:
                    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    img_path = f"{SAVE_DIR}/{count}.jpg"
                    cv2.imwrite(img_path, gray)
                    count += 1

                    cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), (0, 255, 0), 2)
                    cv2.putText(frame, f"Saved {count}", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Face Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if count >= 100:
            break

cap.release()
cv2.destroyAllWindows()

print("Face capture complete.")
