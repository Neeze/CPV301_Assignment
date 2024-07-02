import cv2
import numpy as np
import joblib
import argparse
from skimage.feature import local_binary_pattern, hog
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def detect_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None, None
    return faces, gray

def normalize_image(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def calculate_lbp_features(image):
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 59))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-7)
    return hist

def calculate_hog_features(image):
    fd = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False)
    return fd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference for Emotion Recognition Model")
    parser.add_argument("--model", required=True, help="Path to the model parameters .pkl file")
    args = parser.parse_args()
    classes = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    try:
        model = joblib.load(args.model)
    except:
        print("Error loading model file")
        exit(1)

    vid = cv2.VideoCapture(0)

    while True:
        ret, frame = vid.read()
        if not ret:
            break

        faces_rect, gray_frame = detect_face(frame)
        if faces_rect is not None and len(faces_rect) > 0:  # Check if faces are detected
            for (x, y, w, h) in faces_rect:
                aligned_face = cv2.resize(gray_frame[y:y+h, x:x+w], (48, 48))
                normalized_face = normalize_image(aligned_face)
                features = []
                features.extend(calculate_lbp_features(normalized_face))
                features.extend(calculate_hog_features(normalized_face))
                if features is not None:
                    features = np.array(features).reshape(1, -1)
                    prediction = model.predict(features)
                    label = classes[prediction[0]]

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
