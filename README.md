# 🛡️ SpamShield AI

### Intelligent Email Spam Detection using Deep Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/JavaScript-Frontend-yellow?style=for-the-badge&logo=javascript&logoColor=black" alt="JavaScript">
</p>

<p align="center">
  <strong>AI-powered email classification system that detects whether an email is Spam or Ham using a GRU-based Deep Learning model.</strong>
</p>

<p align="center">
  <a href="#-features">Features</a> •
  <a href="#-demo">Demo</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-api">API</a> •
  <a href="#-deployment">Deployment</a>
</p>

---

## 📌 Overview

**SpamShield AI** is an end-to-end **Email Spam Detection** application built using **Natural Language Processing (NLP)** and **Deep Learning**.

The system analyzes the content of an email and predicts whether it is:

* 🚨 **Spam** — suspicious, unwanted, promotional, fraudulent, or potentially malicious email
* ✅ **Ham** — legitimate and normal email

The project combines a trained **GRU (Gated Recurrent Unit)** neural network with a **FastAPI REST API** and a modern **HTML, CSS, and JavaScript frontend**.

Users can simply enter an email message, click the analyze button, and receive a prediction along with the probability of the email being Spam or Ham.

---

## ✨ Key Features

| Feature              | Description                                                       |
| -------------------- | ----------------------------------------------------------------- |
| 🧠 GRU Deep Learning | Uses a recurrent neural network designed for sequential text data |
| 📧 Spam Detection    | Identifies potentially unwanted or suspicious emails              |
| ✅ Ham Detection      | Identifies legitimate email messages                              |
| 📊 Confidence Score  | Displays prediction confidence and Spam/Ham probabilities         |
| ⚡ FastAPI Backend    | Provides a fast REST API for real-time predictions                |
| 🎨 Modern Frontend   | Interactive HTML, CSS, and JavaScript user interface              |
| 🧹 NLP Preprocessing | Cleans and prepares email text before prediction                  |
| 🔤 Tokenization      | Converts natural language into numerical sequences                |
| 🔄 Sequence Padding  | Converts variable-length emails into fixed-size model input       |
| 🌐 CORS Support      | Allows frontend and backend communication                         |
| 🚀 Deployment Ready  | Backend can be deployed as a FastAPI Web Service                  |

---

## 🎯 Project Goal

The main goal of this project is to demonstrate how **Deep Learning and NLP** can be used to solve a real-world text classification problem.

The complete pipeline is:

```text
Email Text
    │
    ▼
Text Cleaning
    │
    ▼
NLP Preprocessing
    │
    ▼
Tokenization
    │
    ▼
Sequence Padding
    │
    ▼
GRU Deep Learning Model
    │
    ▼
Probability Prediction
    │
    ├───────────────┐
    ▼               ▼
  SPAM            HAM
    │               │
    └───────┬───────┘
            ▼
     Result + Confidence
```

---

# 🧠 Machine Learning Model

The core of this project is a **GRU-based Deep Learning model**.

### Why GRU?

GRU, or **Gated Recurrent Unit**, is a type of Recurrent Neural Network (RNN) that is well suited for sequential data such as natural language.

For email classification, the order and context of words can provide important information.

For example:

```text
"Congratulations! You won a free prize!"
```

contains different patterns from:

```text
"Hi team, our meeting is scheduled for tomorrow."
```

The GRU model learns these patterns from the training data and uses them to classify new emails.

---

## 🔄 Prediction Pipeline

```text
                 USER EMAIL
                     │
                     ▼
          ┌────────────────────┐
          │   FastAPI Request  │
          └─────────┬──────────┘
                    │
                    ▼
          ┌────────────────────┐
          │ Text Preprocessing │
          │                    │
          │ • Lowercase        │
          │ • Cleaning         │
          │ • Tokenization     │
          │ • Stopwords        │
          └─────────┬──────────┘
                    │
                    ▼
          ┌────────────────────┐
          │ Saved Tokenizer    │
          │ tokenizer.pickle   │
          └─────────┬──────────┘
                    │
                    ▼
          ┌────────────────────┐
          │ Sequence Padding   │
          └─────────┬──────────┘
                    │
                    ▼
          ┌────────────────────┐
          │    GRU Model       │
          │ gru_model.keras    │
          └─────────┬──────────┘
                    │
                    ▼
          ┌────────────────────┐
          │ Probability Score  │
          └─────────┬──────────┘
                    │
             ┌──────┴──────┐
             ▼             ▼
          🚨 SPAM        ✅ HAM
```

---

# 📊 Prediction Example

### 📩 Input Email

```text
Congratulations! You have won a free prize.
Click this link to claim your reward immediately.
```

### 🤖 Model Response

```json
{
  "prediction": "Spam",
  "confidence": 0.9998,
  "probability_spam": 0.9998,
  "probability_ham": 0.0002
}
```

### 📈 Result

```text
🚨 SPAM DETECTED

Spam Probability : 99.98%
Ham Probability  : 0.02%
Confidence       : 99.98%
```

---

# 🖥️ Application Preview

> 📸 Add your application screenshot here.

```text
frontend/
    │
    ├── index.html
    ├── style.css
    └── script.js
```

You can add a screenshot to your GitHub repository and display it like this:

```markdown
![SpamShield AI Dashboard](screenshots/dashboard.png)
```

Recommended folder:

```text
screenshots/
└── dashboard.png
```

---

# 🏗️ System Architecture

```text
                     ┌───────────────────┐
                     │       USER        │
                     └─────────┬─────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │     WEB FRONTEND       │
                  │                        │
                  │ HTML + CSS + JavaScript│
                  └────────────┬───────────┘
                               │
                         HTTP POST
                               │
                               ▼
                  ┌────────────────────────┐
                  │      FASTAPI API       │
                  │                        │
                  │      /predict          │
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │  TEXT PREPROCESSING    │
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │  TOKENIZER + PADDING   │
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │      GRU MODEL         │
                  │    gru_model.keras      │
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │    PREDICTION RESULT   │
                  │                        │
                  │    Spam / Ham          │
                  │    Probability         │
                  │    Confidence          │
                  └────────────┬───────────┘
                               │
                               ▼
                  ┌────────────────────────┐
                  │      WEB FRONTEND      │
                  │                        │
                  │    Display Result      │
                  └────────────────────────┘
```

---

# 🛠️ Technology Stack

### 🧠 Artificial Intelligence

* Python
* TensorFlow
* Keras
* GRU
* Natural Language Processing

### 🧹 Data Processing

* NumPy
* NLTK

### ⚡ Backend

* FastAPI
* Uvicorn
* Pydantic

### 🎨 Frontend

* HTML5
* CSS3
* JavaScript
* Fetch API

### 🔧 Tools

* Jupyter Notebook
* Git
* GitHub
* Visual Studio Code

---

# 📁 Project Structure

```text
Email-Spam-Detection/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── main.py
├── requirements.txt
├── runtime.txt
│
├── gru_model.keras
├── tokenizer.pickle
├── config.pkl
├── label_mapping.pkl
│
├── model.ipynb
├── .gitignore
└── README.md
```

### File Description

| File                  | Purpose                                     |
| --------------------- | ------------------------------------------- |
| `main.py`             | FastAPI backend and prediction endpoint     |
| `gru_model.keras`     | Trained GRU Deep Learning model             |
| `tokenizer.pickle`    | Saved text tokenizer                        |
| `config.pkl`          | Saved model configuration                   |
| `label_mapping.pkl`   | Saved class label mapping                   |
| `requirements.txt`    | Python dependencies                         |
| `runtime.txt`         | Python runtime version for deployment       |
| `model.ipynb`         | Model training and experimentation notebook |
| `frontend/index.html` | Application user interface                  |
| `frontend/style.css`  | UI styling and animations                   |
| `frontend/script.js`  | Frontend logic and API communication        |

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Soumendra-Barick/Email-Spam-Detection.git
```

Move into the project directory:

```bash
cd Email-Spam-Detection
```

---

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv venv
```

Activate the environment:

```powershell
.\venv\Scripts\Activate.ps1
```

If using Command Prompt:

```cmd
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Download NLTK Resources

If your application uses NLTK tokenization and stopwords, run:

```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('stopwords')"
```

---

# ▶️ Run the Backend

Start the FastAPI application:

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

---

# 📚 API Documentation

FastAPI automatically generates interactive API documentation.

Open:

```text
http://127.0.0.1:8000/docs
```

You can test the API directly using the Swagger UI.

---

# 🔌 API Usage

## Endpoint

```text
POST /predict
```

### Request

```json
{
  "email": "Congratulations! You have won a free prize. Click this link to claim your reward."
}
```

### Response

```json
{
  "email": "Congratulations! You have won a free prize. Click this link to claim your reward.",
  "prediction": "Spam",
  "confidence": 0.9998,
  "probability_spam": 0.9998,
  "probability_ham": 0.0002
}
```

---

# 🌐 Run the Frontend

The frontend is located inside:

```text
frontend/
```

Open the folder using Visual Studio Code and run the application using **Live Server**.

The frontend will typically be available at:

```text
http://127.0.0.1:5500
```

The frontend sends the email to the FastAPI backend:

```text
Frontend
    │
    │ POST /predict
    ▼
http://127.0.0.1:8000/predict
```

---

# 🌍 CORS Configuration

During local development, the frontend and backend run on different ports:

```text
Frontend
http://127.0.0.1:5500

Backend
http://127.0.0.1:8000
```

Therefore, FastAPI requires CORS configuration to allow communication between them.

For production deployment, update the allowed origins with your actual frontend deployment URL.

---

# 🚀 Deployment

The FastAPI backend can be deployed using **Render** or another cloud platform that supports Python web services.

### Render Build Command

```bash
pip install -r requirements.txt
```

### Render Start Command

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

After deployment, your API will be available at a URL similar to:

```text
https://your-app-name.onrender.com
```

The API documentation will be available at:

```text
https://your-app-name.onrender.com/docs
```

### ⚠️ Important

After deploying the backend, update the frontend JavaScript API URL.

Local:

```javascript
fetch("http://127.0.0.1:8000/predict")
```

Production:

```javascript
fetch("https://your-app-name.onrender.com/predict")
```

Also update the FastAPI CORS configuration to allow your deployed frontend URL.

---

# 📦 Dataset

The original training dataset is not included in this GitHub repository because the dataset file is approximately **140 MB**, exceeding GitHub's standard 100 MB per-file limit.

The trained model and preprocessing artifacts required for inference are included in the repository.

```text
Training Dataset
      │
      ▼
Data Preprocessing
      │
      ▼
GRU Model Training
      │
      ▼
Saved Model
      │
      ├── gru_model.keras
      ├── tokenizer.pickle
      ├── config.pkl
      └── label_mapping.pkl
      │
      ▼
FastAPI Prediction API
```

---

# 🧪 Example Emails

## 🚨 Spam Example

```text
Congratulations! You have won a $1,000 cash prize.
You are one of our lucky winners. Click the link below
to claim your reward immediately.
```

## ✅ Ham Example

```text
Hi Team,

This is a reminder that our project meeting is scheduled
for tomorrow at 10 AM. Please bring your latest project
updates.

Regards
```

---

# 🔮 Future Improvements

* [ ] Deploy the frontend
* [ ] Connect production frontend with FastAPI backend
* [ ] Add user authentication
* [ ] Add email prediction history
* [ ] Add prediction analytics
* [ ] Add batch email classification
* [ ] Improve model accuracy
* [ ] Compare GRU with LSTM and BiLSTM
* [ ] Implement Transformer-based classification
* [ ] Add phishing URL detection
* [ ] Add suspicious link analysis
* [ ] Add attachment analysis
* [ ] Add Docker support
* [ ] Add CI/CD pipeline

---

# 🔐 Disclaimer

This project is developed for **educational and demonstration purposes**.

The prediction generated by this system should not be considered a complete cybersecurity solution. A production-grade email security system should combine machine learning with additional techniques such as:

* Sender reputation analysis
* URL scanning
* Phishing detection
* Attachment scanning
* Domain verification
* Malware detection
* Email header analysis

---

# 👨‍💻 Author

## Soumendra Barick

Machine Learning & Deep Learning Enthusiast

This project demonstrates the integration of:

```text
Machine Learning
      +
Deep Learning
      +
Natural Language Processing
      +
FastAPI
      +
Frontend Development
      +
Cloud Deployment
```

---

# ⭐ Support

If you found this project useful or interesting, consider giving the repository a ⭐ on GitHub.

Your support is greatly appreciated!

---

<p align="center">
  Made with ❤️ using Python, TensorFlow, FastAPI and JavaScript
</p>
