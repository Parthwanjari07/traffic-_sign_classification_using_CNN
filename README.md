# 🚦 Traffic Sign Classification Using CNN

This project implements a Convolutional Neural Network (CNN) to accurately classify traffic signs, enhancing road safety and aiding autonomous driving systems.

---

## 🚀 Features

- **Deep Learning Model**: CNN architecture trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset to classify 43 traffic sign categories.
- **Interactive GUI**: User-friendly interface built with Streamlit for real-time predictions.
- **Pre-trained Models**: Includes pre-trained `.h5` models for instant inference.

---

## 📁 Project Structure

├── app.py ├── gui.py ├── StremelitGui.py ├── traffic_sign.py ├── my_model.h5 ├── traffic_classifier.h5 ├── requirements.txt ├── .gitignore ├── .gitattributes ├── templates/ └── README.md

markdown
Copy
Edit

- `app.py`: Main application logic.
- `gui.py` & `StremelitGui.py`: GUI interfaces for interaction.
- `traffic_sign.py`: Core classification utilities.
- `*.h5`: Pre-trained model weights.
- `requirements.txt`: Dependency list.
- `templates/`: HTML template directory.

---

## 🧠 Model Overview

The CNN model typically includes:

- Convolutional layers with ReLU activation
- Max Pooling layers
- Dropout layers for regularization
- Fully connected (dense) layers
- Softmax output for multi-class classification (43 categories)

---

## 🖥️ Installation & Usage

### 🔧 Prerequisites

- Python 3.6+

### 📦 Setup

1. **Clone the repo**

```bash
git clone https://github.com/Parthwanjari07/traffic-_sign_classification_using_CNN.git
cd traffic-_sign_classification_using_CNN
(Optional) Create a virtual environment

bash
Copy
Edit
python -m venv venv
# Activate the environment:
# On Windows:
venv\Scripts\activate
# On Unix/Mac:
source venv/bin/activate
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
🚀 Run the Application
To launch the GUI with Streamlit:

bash
Copy
Edit
streamlit run StremelitGui.py
📊 Dataset
This project uses the GTSRB Dataset — over 50,000 labeled images in 43 classes of German road signs. It is one of the most widely used datasets in traffic sign classification research.

📈 Model Performance
The model achieves high accuracy on the GTSRB dataset (specific metrics may vary). Typical CNNs on this dataset reach >95% validation accuracy.

🤝 Contribution
Contributions are welcome! Here's how you can help:

Fork this repo

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add AmazingFeature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📄 License
This project is licensed under the MIT License.

