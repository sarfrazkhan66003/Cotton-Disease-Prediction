# ===============================
# Cotton Disease Prediction Model
# ===============================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import os

# Set paths
train_dir = r"C:\Users\DELL\Desktop\Sarfu\PW Data Science\Project\Cotton-Disease-Prediction-Deep-Learning-master\dataset\train"
val_dir = r"C:\Users\DELL\Desktop\Sarfu\PW Data Science\Project\Cotton-Disease-Prediction-Deep-Learning-master\dataset\val"

# ============================
# Data Preprocessing
# ============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# ============================
# Build ResNet152V2 Model
# ============================
base_model = ResNet152V2(weights='imagenet', include_top=False, input_shape=(224,224,3))

for layer in base_model.layers:
    layer.trainable = False  # freeze base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ============================
# Compile Model
# ============================
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ============================
# Train Model
# ============================
EPOCHS = 15

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# ============================
# Save Model
# ============================
model.save("model_resnet152V2.h5")
print("✅ Model saved as model_resnet152V2.h5")
