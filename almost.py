import cv2
import dlib
import numpy as np
from scipy.spatial.distance import cosine
import os
import threading
import queue

try:
    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
    print("Models loaded successfully.")
except Exception as e:
    print("Error in loading models:", e)


dataset_folder = "dataset"
similarity_percentage= 95
encodings_cache = {}


def preprocess_image(image):
    return np.ascontiguousarray(image.astype(np.uint8))

def detect_faces(image):
    return face_detector(image)

def extract_face(image, face):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    shape = shape_predictor(image_rgb, face)
    return np.array(face_recognition_model.compute_face_descriptor(image_rgb, shape))

def check_similarity(encoding1, encoding2):
    return (1 - cosine(encoding1, encoding2)) * 100

def load_dataset_encodings():
    for file_name in os.listdir(dataset_folder):
        file_path = os.path.join(dataset_folder, file_name)
        dataset_image = cv2.imread(file_path)
        if dataset_image is None:
            print(f"Warning: Unable to read image {file_path}")
            continue
        dataset_image = preprocess_image(dataset_image)
        faces_in_dataset = detect_faces(dataset_image)
        
        if len(faces_in_dataset) > 0:
            dataset_encoding = extract_face(dataset_image, faces_in_dataset[0])
            encodings_cache[file_name] = dataset_encoding
        else:
            print(f"No faces detected in {file_name}")

def match_with_dataset(encoding_captured):
    for file_name, dataset_encoding in encodings_cache.items():
        similarity = check_similarity(encoding_captured, dataset_encoding)
        
        if similarity > similarity_percentage:
            return file_name, similarity
    return None, 0

def capture_frames(cap, frame_queue):
    while True:
        ret, frame = cap.read()
        if ret:
            if frame_queue.full():
                frame_queue.get()  
            frame_queue.put(frame)

def capture_and_recognize():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam could not be opened. Check your camera.")

    load_dataset_encodings()  

    frame_queue = queue.Queue(maxsize=0.0001)
    capture_thread = threading.Thread(target=capture_frames, args=(cap, frame_queue))
    capture_thread.daemon = True
    capture_thread.start()
    try:
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                frame = preprocess_image(frame)
                faces = detect_faces(frame)

                if len(faces) == 0:
                    print("No faces detected.")
                for face in faces:
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    encoding_captured = extract_face(frame, face)
                    name, similarity = match_with_dataset(encoding_captured)
                    if similarity > similarity_percentage:
                        color = (0, 255, 0)  
                        cv2.putText(frame, f"Welcome{name} ({similarity:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        print("Door Open")  
                    else:
                        color = (0, 0, 255)  
                        cv2.putText(frame, f"Stranger", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        print("Get Out")  
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                cv2.imshow("Webcam", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_recognize()



