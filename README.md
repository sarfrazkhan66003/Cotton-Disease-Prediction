🧠 Cotton Disease Prediction using Deep Learning

This project is an AI-powered web application that detects diseases in cotton plants from leaf images using a ResNet152V2 deep learning model.
It helps farmers and researchers identify plant health conditions early, preventing crop loss and improving agricultural productivity. 🌿

🚀 Project Overview

The system takes an image of a cotton leaf or plant as input and classifies it into one of the following categories:
1️⃣ Diseased Cotton Leaf
2️⃣ Diseased Cotton Plant
3️⃣ Fresh Cotton Leaf
4️⃣ Fresh Cotton Plant
It uses a Convolutional Neural Network (CNN) based on ResNet152V2 for image classification, trained on a custom dataset of cotton plant images.

📁 Project Structure

Cotton-Disease-Prediction/
│
├── app.py                     # Flask web app
├── model_resnet152V2.h5       # Trained CNN model
├── templates/
│   └── index.html             # Frontend page
├── uploads/                   # Folder for uploaded test images
├── static/                    # Static files (CSS/JS/images)
├── requirements.txt           # Dependencies
└── README.md                  # Documentation

🧠 Classes Predicted

| Label | Description           |
| ----- | --------------------- |
| 🧪 0  | Diseased Cotton Leaf  |
| 🌿 1  | Diseased Cotton Plant |
| 🍃 2  | Fresh Cotton Leaf     |
| 🌱 3  | Fresh Cotton Plant    |


🧩 Features

✅ Deep Learning Model (ResNet152V2) — fine-tuned on cotton leaf/plant images for high accuracy
✅ Flask Web Application — simple and interactive UI for predictions
✅ Real-time Image Upload — upload any cotton image and get instant results
✅ Automated Preprocessing — image normalization, resizing, and prediction pipeline
✅ Model Deployment Ready — can be hosted locally or on platforms like Render, HuggingFace


🧠 Algorithm and Model Architecture

The model is built on Transfer Learning using the ResNet152V2 architecture.
🔹 Steps Involved:

  1.Data Preprocessing
    Image resizing to (224×224)
    Normalization (scaling pixel values between 0 and 1)
    One-hot encoding of labels
  
  2.Model Architecture
    Base Model: ResNet152V2 (pretrained on ImageNet)
    Added layers:
      Global Average Pooling
      Dense (512, ReLU)
      Dropout (0.5)
      Output Layer (4 neurons, Softmax)
  
  3.Training
    Optimizer: Adam
    Loss Function: Categorical Crossentropy
    Epochs: 20
    Early stopping and model checkpointing for best performance
    
  4.Deployment
    Saved as model_resnet152V2.h5
    Integrated into Flask app (app.py)

    

⚙️ Tech Stack
| Category                  | Tools / Libraries                   |
| ------------------------- | ----------------------------------- |
| **Language**              | Python 3.x                          |
| **Framework**             | TensorFlow, Keras                   |
| **Web App**               | Flask                               |
| **Frontend**              | HTML, CSS                           |
| **Model**                 | ResNet152V2                         |
| **IDE/Environment**       | Jupyter Notebook, VS Code, Anaconda |
| **Deployment (Optional)** | Render / Heroku / AWS EC2           |



📊 Output

🌱 Input:
An image of a cotton leaf or plant (healthy or diseased).
🔍 Output:
Predicted class:
  The leaf is diseased cotton leaf
  The leaf is diseased cotton plant
  The leaf is fresh cotton leaf
  The leaf is fresh cotton plant

