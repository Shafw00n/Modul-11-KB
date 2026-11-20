import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import os

# Set seed agar hasil stabil dan persis dengan modul
tf.random.set_seed(0)
np.random.seed(0)

# PARAMETER
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
dataset = "Dataset"

# DATASET
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)

train_data = datagen.flow_from_directory(
    dataset,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True,
    seed=0
)

validation_data = datagen.flow_from_directory(
    dataset,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False,
    seed=0
)

# INFORMASI DATA
total_data = train_data.samples + validation_data.samples
print("====================================")
print("JUMLAH DATA")
print("====================================")
print("Total data :", total_data)
print("Data training :", train_data.samples)
print("Data validasi :", validation_data.samples)
print("Training :", round(train_data.samples/total_data*100, 2), "%")
print("Validasi :", round(validation_data.samples/total_data*100, 2), "%")
print("Class mapping :", train_data.class_indices)

# MODEL CNN
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

# COMPILE
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# TRAINING
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS
)

# GRAFIK ACCURACY DAN LOSS
plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(history.history["accuracy"], label="Training")
plt.plot(history.history["val_accuracy"], label="Validasi")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(history.history["loss"], label="Training")
plt.plot(history.history["val_loss"], label="Validasi")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("grafik_training.png")
try:
    plt.show(block=False)
except Exception:
    pass
plt.close()

# EVALUASI
loss, accuracy = model.evaluate(validation_data)
print("====================================")
print("HASIL VALIDASI")
print("====================================")
print("Validation Loss :", round(loss, 4))
print("Validation Accuracy :", round(accuracy*100, 2), "%")

# PREDIKSI VALIDASI
validation_data.reset()
hasil = model.predict(validation_data)
y_pred = (hasil >= 0.5).astype(int).reshape(-1)
y_true = validation_data.classes

# CONFUSION MATRIX
cm = confusion_matrix(y_true, y_pred)
print("====================================")
print("CONFUSION MATRIX")
print("====================================")
print(cm)

# CLASSIFICATION REPORT
nama_kelas = list(validation_data.class_indices.keys())
print("====================================")
print("CLASSIFICATION REPORT")
print("====================================")
print(classification_report(y_true, y_pred, target_names=nama_kelas))

# SIMPAN MODEL
model.save("model_motor_mobil.keras")
