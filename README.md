# Image Classification using CNN in TensorFlow

This project demonstrates image classification using a Convolutional Neural Network (CNN) implemented in TensorFlow and Keras. It loads an image dataset, preprocesses it, builds a CNN model, trains it, and evaluates its performance using accuracy metrics and a confusion matrix.

## Features
- Image loading and preprocessing using TensorFlow's `image_dataset_from_directory`
- Splitting the dataset into training, validation, and test sets
- CNN model architecture for multi-class image classification
- Model evaluation using classification reports and confusion matrices
- Visualization of training performance and dataset images

## Requirements
Before running the project, ensure you have the following dependencies installed:

```bash
pip install tensorflow numpy pandas seaborn matplotlib scikit-learn
```

## Dataset
The dataset should be placed inside the `/content/dataset` directory. Ensure that the dataset follows the folder structure:
```
dataset/
    class_1/
        image1.jpg
        image2.jpg
    class_2/
        image1.jpg
        image2.jpg
    ...
```

## Running the Project
1. **Mount Google Drive (if necessary):**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Load and preprocess the dataset:**
   ```python
   data_dir = pathlib.Path("/content/dataset")
   data = image_dataset_from_directory(data_dir, seed=123, image_size=(224, 224))
   ```

3. **Visualize sample images:**
   ```python
   plt.figure(figsize=(10, 10))
   for images, labels in data.take(1):
       for i in range(9):
           ax = plt.subplot(3, 3, i + 1)
           plt.imshow(images[i].numpy().astype("uint8"))
           plt.title(data.class_names[labels[i]])
           plt.axis("off")
   ```

4. **Split the dataset:**
   ```python
   train_size = int(0.7 * len(data)) + 1
   val_size = int(0.2 * len(data))
   test_size = int(0.1 * len(data))

   train = data.take(train_size)
   remaining = data.skip(train_size)
   val = remaining.take(val_size)
   test = remaining.skip(val_size)
   ```

5. **Build and compile the CNN model:**
   ```python
   model = Sequential([
       Conv2D(64, (3,3), activation='relu', padding='same', input_shape=(224,224,3)),
       Conv2D(64, (3,3), activation='relu', padding='same'),
       Conv2D(64, (3,3), activation='relu', padding='same'),
       MaxPool2D(),
       Conv2D(128, (3,3), activation='relu', padding='same'),
       Conv2D(128, (3,3), activation='relu', padding='same'),
       Conv2D(128, (3,3), activation='relu', padding='same'),
       MaxPool2D(),
       Flatten(),
       Dense(256, activation='relu'),
       Dense(4, activation='softmax')
   ])
   model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   model.summary()
   ```

6. **Train the model:**
   ```python
   history = model.fit(train, validation_data=val, epochs=10)
   ```

7. **Plot model performance:**
   ```python
   def plot_performance(epochs, history):
       acc = history.history['accuracy']
       val_acc = history.history['val_accuracy']
       loss = history.history['loss']
       val_loss = history.history['val_loss']
       
       plt.figure(figsize=(8, 8))
       plt.subplot(1, 2, 1)
       plt.plot(range(epochs), acc, label='Training Accuracy')
       plt.plot(range(epochs), val_acc, label='Validation Accuracy')
       plt.legend(loc='lower right')
       plt.title('Training and Validation Accuracy')
       
       plt.subplot(1, 2, 2)
       plt.plot(range(epochs), loss, label='Training Loss')
       plt.plot(range(epochs), val_loss, label='Validation Loss')
       plt.legend(loc='upper right')
       plt.title('Training and Validation Loss')
       plt.show()
   ```

8. **Evaluate the model:**
   ```python
   def evaluate_model(model):
       model.evaluate(test)
       y_pred = np.argmax(model.predict(test_set['images']), 1)
       print(classification_report(y_test, y_pred, target_names=class_names))
       cm = confusion_matrix(y_test, y_pred)
       plt.figure(figsize=(10, 8))
       sn.heatmap(cm, annot=True)
       plt.xlabel("Predicted")
       plt.ylabel("Actual")
       plt.title("Confusion Matrix")
   ```

9. **Run the evaluation function:**
   ```python
   evaluate_model(model)
   ```

## Results
- Displays sample images from the dataset
- Plots training and validation accuracy/loss
- Generates a classification report and confusion matrix

## Enhancements for Real-Life Applications
This model can be enhanced and adapted to solve real-world problems, such as:
- **Medical Image Classification**: Using transfer learning with pre-trained models (e.g., ResNet, EfficientNet) to detect diseases in X-rays, MRIs, or CT scans.
- **Automated Quality Control**: Implementing object detection techniques to identify defective products in manufacturing.
- **Wildlife Conservation**: Using image classification to identify and track endangered species in camera trap images.
- **Security & Surveillance**: Enhancing the model with object detection and anomaly detection for automated threat detection.
- **Retail & Inventory Management**: Applying the model to recognize and categorize products for automated checkout or stock management.
- **Autonomous Vehicles**: Integrating the model into self-driving systems for real-time object recognition and hazard detection.

## Acknowledgments
- TensorFlow/Keras for deep learning framework
- Scikit-learn for evaluation metrics
- Seaborn and Matplotlib for visualization

---
This project is ideal for understanding CNN-based image classification and can be extended with additional optimizations such as data augmentation, hyperparameter tuning, and advanced architectures like ResNet or EfficientNet.

