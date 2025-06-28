import os
import cv2
import numpy as np

from keras.models import load_model #type: ignore

model = load_model('ulcer_classifier_simple.keras')

test_folder = "C:/Users/sneha/OneDrive/Documents/python/autoencoders/Patches/Normal(Healthy skin)"

for file in os.listdir(test_folder):
    path = os.path.join(test_folder, file)
    img = cv2.imread(path)
    if img is None:
        continue

    # Resize and normalize
    resized = cv2.resize(img, (128,128)).astype(np.float32) / 255.0
    input_img = np.expand_dims(resized, axis=0)

    # Predict
    pred = model.predict(input_img)[0][0]
    label = "Ulcer" if pred >= 0.5 else "Normal"
    prob = f"{pred:.2f}"

    # Add label text to original (not-resized) image
    display_img = img.copy()
    text = f"{label} ({prob})"
    color = (0, 0, 255) if label == "Ulcer" else (0, 255, 0)  # red for ulcer, green for normal
    cv2.putText(display_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Show image with prediction
    cv2.imshow("Prediction", display_img)
    key = cv2.waitKey(0)  # Wait for a key press
    if key == 27:  # Press Esc to break
        break

cv2.destroyAllWindows()