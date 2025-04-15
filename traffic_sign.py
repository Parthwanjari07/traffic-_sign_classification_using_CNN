import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import seaborn as sns

data = []
labels = []
classes = 43
# Update base path to point to the Dataset directory
base_path = os.path.join(os.path.dirname(os.getcwd()), '..', 'Dataset')

#Retrieving the images and their labels 
for i in range(classes):
    path = os.path.join(base_path, 'Train', str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(os.path.join(path, a))
            image = image.resize((30,30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print(f"Error loading image: {os.path.join(path, a)}")

#Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

# Data Analysis Section
print("\n=== Dataset Analysis ===")
print(f"Total number of images: {len(data)}")
print(f"Number of classes: {classes}")

# Display distribution of classes
unique, counts = np.unique(labels, return_counts=True)
plt.figure(figsize=(15, 5))
plt.bar(unique, counts)
plt.title('Distribution of Traffic Sign Classes')
plt.xlabel('Class ID')
plt.ylabel('Number of Images')
plt.show()

# Display sample images from different classes
plt.figure(figsize=(15, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    idx = np.random.randint(0, len(data))
    plt.imshow(data[idx])
    plt.title(f'Class {labels[idx]}')
    plt.axis('off')
plt.show()

# Basic image statistics
print("\nImage Statistics:")
print(f"Image dimensions: {data[0].shape}")
print(f"Min pixel value: {data.min()}")
print(f"Max pixel value: {data.max()}")
print(f"Mean pixel value: {data.mean():.2f}")
print(f"Standard deviation: {data.std():.2f}")

# Advanced Data Analysis
print("\n=== Advanced Dataset Analysis ===")

# 1. Enhanced Class Distribution Analysis
plt.figure(figsize=(15, 6))
sns.barplot(x=unique, y=counts)
plt.title('Enhanced Distribution of Traffic Sign Classes')
plt.xlabel('Class ID')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
# Add count labels on top of each bar
for i, v in enumerate(counts):
    plt.text(i, v, str(v), ha='center', va='bottom')
plt.tight_layout()
plt.show()

# 2. Class Balance Analysis
print("\nClass Balance Analysis:")
min_samples = min(counts)
max_samples = max(counts)
print(f"Minimum samples per class: {min_samples} (Class {unique[np.argmin(counts)]})")
print(f"Maximum samples per class: {max_samples} (Class {unique[np.argmax(counts)]})")
print(f"Imbalance ratio: {max_samples/min_samples:.2f}:1")

# 3. Color Channel Analysis
plt.figure(figsize=(15, 5))
for i, channel in enumerate(['Red', 'Green', 'Blue']):
    plt.subplot(1, 3, i+1)
    channel_data = data[:, :, :, i].ravel()
    plt.hist(channel_data, bins=50, color=['red', 'green', 'blue'][i], alpha=0.7)
    plt.title(f'{channel} Channel Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 4. Image Quality Assessment
print("\nImage Quality Metrics:")
# Calculate average brightness and contrast
brightness = np.mean(data, axis=(1,2,3))
contrast = np.std(data, axis=(1,2,3))
print(f"Average brightness across dataset: {np.mean(brightness):.2f}")
print(f"Average contrast across dataset: {np.mean(contrast):.2f}")

# 5. Sample Images with Augmentation Preview
plt.figure(figsize=(15, 8))
for i in range(3):
    # Original image
    idx = np.random.randint(0, len(data))
    plt.subplot(2, 3, i+1)
    plt.imshow(data[idx])
    plt.title(f'Original (Class {labels[idx]})')
    plt.axis('off')
    
    # Augmented version (simple rotation)
    plt.subplot(2, 3, i+4)
    augmented = np.rot90(data[idx])
    plt.imshow(augmented)
    plt.title('Rotated 90°')
    plt.axis('off')
plt.suptitle('Original vs Augmented Samples')
plt.tight_layout()
plt.show()

# 6. Dimensionality Analysis
print("\nDimensionality Analysis:")
print(f"Total dataset size in memory: {data.nbytes / (1024 * 1024):.2f} MB")
print(f"Number of features per image: {np.prod(data[0].shape)}")

# Now proceed with train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#Converting the labels into one hot encoding
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

#Building the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

#Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 15
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))
model.save("my_model.h5")

#plotting graphs for accuracy 
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

#testing accuracy on test dataset
from sklearn.metrics import accuracy_score

# Update test data path
test_csv_path = os.path.join(base_path, 'Test.csv')
y_test = pd.read_csv(test_csv_path)

labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data=[]

for img in imgs:
    # Update image path to use base_path
    image_path = os.path.join(base_path, img)
    image = Image.open(image_path)
    image = image.resize((30,30))
    data.append(np.array(image))

X_test=np.array(data)

pred = model.predict_classes(X_test)

#Accuracy with the test data
from sklearn.metrics import accuracy_score
print(accuracy_score(labels, pred))
