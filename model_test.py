import tensorflow as tf
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ========================
# Dataset paths
# ========================
base_dir = r"C:\Users\DELL\Desktop\Sarfu\PW Data Science\Project\Cotton-Disease-Prediction-Deep-Learning-master\Datasets"
train_dir = os.path.join(base_dir, r"C:\Users\DELL\Desktop\Sarfu\PW Data Science\Project\Cotton-Disease-Prediction-Deep-Learning-master\dataset\train")
val_dir = os.path.join(base_dir, r"C:\Users\DELL\Desktop\Sarfu\PW Data Science\Project\Cotton-Disease-Prediction-Deep-Learning-master\dataset\val")

# ========================
# Data generators
# ========================
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

validation_set = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

# ========================
# Build ResNet152V2 model
# ========================
base_model = ResNet152V2(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Freeze the base model
for layer in base_model.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)  # ✅ 4 output classes

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ========================
# Train the model
# ========================
history = model.fit(
    training_set,
    validation_data=validation_set,
    epochs=20,
    steps_per_epoch=len(training_set),
    validation_steps=len(validation_set)
)

# ========================
# Save the model
# ========================
model.save("model_resnet152V2.h5")
print("✅ Model saved successfully.")
