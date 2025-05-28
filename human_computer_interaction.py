import numpy as np
import pandas as pd
import os 
import cv2
import random
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Dropout,Flatten, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import warnings 
warnings.filterwarnings('ignore')

#########################################################################################################################



path = r"C:/Users/PC/Desktop/leapGestRecog"


images = []
labels = []

for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)
    if os.path.isdir(folder_path):
        for subfolder in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder)
            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(subfolder_path, file)
                        images.append(file_path)
                        labels.append(subfolder)


random_indices = random.sample(range(len(images)), 20)
random_images = [images[i] for i in random_indices]
random_labels = [labels[i] for i in random_indices]

fig, axes = plt.subplots(5, 4, figsize=(16, 10))
axes = axes.flatten()
for idx, (img_path, label) in enumerate(zip(random_images, random_labels)):
    img = Image.open(img_path)
    axes[idx].imshow(img)
    axes[idx].set_title(label)
    axes[idx].axis('off')
plt.tight_layout()
plt.show()


df = pd.DataFrame({'images': images, 'labels': labels})


le = LabelEncoder()
df['labels'] = le.fit_transform(df['labels'])


x = []
img_size = (64, 64) 

for img_path in df['images']:
    img = Image.open(img_path).convert('RGB')
    img_resized = img.resize(img_size)
    img_array = np.array(img_resized) / 255.0  
    x.append(img_array)

x = np.array(x)
y = np.array(df['labels'])
y = to_categorical(y)

print(f"x shape: {x.shape}")
print(f"y shape: {y.shape}")
print(f"Number of classes: {len(le.classes_)}")

#################################################################################################################################

# train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=54)

# CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)


history = model.fit(x_train, y_train, epochs=10, batch_size=32,
                    validation_data=(x_test, y_test), callbacks=[early_stopping])


print("*MODEL SUMMARY*")
model.summary()


print("\nTEST RESULTS")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")


predictions = model.predict(x_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

print(f"Number of predicted classes : {len(np.unique(predicted_classes))}")
print(f"Number of actual classes    : {len(np.unique(true_classes))}")

print("\nCLASSIFICATION REPORT")
print(classification_report(true_classes, predicted_classes))


random_indices = random.sample(range(len(x_test)), 10)
fig, axes = plt.subplots(5, 2, figsize=(10, 20))
axes = axes.flatten()

for i, idx in enumerate(random_indices):
    img = x_test[idx]
    actual_class = true_classes[idx]
    predicted_class = predicted_classes[idx]

    actual_label = le.inverse_transform([actual_class])[0]
    predicted_label = le.inverse_transform([predicted_class])[0]

    axes[i].imshow(img)
    axes[i].axis('off')

  
    axes[i].text(0, img.shape[0] + 5, f"Predicted: {predicted_label}", fontsize=9, color='white', backgroundcolor='black')
    axes[i].text(0, img.shape[0] + 18, f"Actual: {actual_label}", fontsize=9, color='white', backgroundcolor='black')

plt.tight_layout()
plt.show()




plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()




###########################################################################################
model.save("gesture_model.h5")














 

















