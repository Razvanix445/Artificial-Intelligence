import numpy as np
import cv2
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from CNN_manual.CNN import CNNModel


def predict_image(model, image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Resize the image
    image = cv2.resize(image, (64, 64))

    # Use the trained model to predict the class of the image
    prediction = model.predict(image)

    # Return the predicted class
    return "Normal" if prediction[0] == 0 else "Sepia"

def CNN_manual(real_images_paths, sepia_images_paths):
    # Step 1: Load the images
    real_images = []
    for path in real_images_paths:
        for file in glob.glob(path + "/*.jpg"):
            img = cv2.imread(file)
            if img is not None:
                real_images.append(cv2.resize(img, (64, 64)))

    sepia_images = []
    for path in sepia_images_paths:
        for file in glob.glob(path + "/*.jpg"):
            img = cv2.imread(file)
            if img is not None:
                sepia_images.append(cv2.resize(img, (64, 64)))

    # Step 2: Convert the images into a format that can be used by the CNN model
    real_images = np.array(real_images)
    sepia_images = np.array(sepia_images)
    data = np.vstack([real_images, sepia_images])
    labels = np.array([0] * len(real_images) + [1] * len(sepia_images))

    # Step 3: Split the data into training and testing sets
    trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Step 4: Instantiate and train the CNN model
    input_shape = (64, 64, 3)  # Assuming your input images are 64x64 pixels with 3 color channels (RGB)
    num_classes = 2  # Assuming you have 2 classes: normal and sepia
    model = CNNModel(input_shape, num_classes)
    model.train(trainX, trainY, learning_rate=0.001, epochs=10)  # Adjust learning rate and epochs as needed

    # Step 5: Evaluate the CNN model
    test_predictions = model.predict(testX)
    accuracy = accuracy_score(testY, test_predictions)
    precision = precision_score(testY, test_predictions, average='weighted', zero_division=1)
    recall = recall_score(testY, test_predictions, average='weighted', zero_division=1)
    f1 = f1_score(testY, test_predictions, average='weighted', zero_division=1)

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')