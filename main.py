# ============================================================
# SPAMSHIELD AI - FASTAPI BACKEND
# Email Spam / Ham Detection using GRU Deep Learning
# ============================================================

import os
import pickle
import string
import re
import time

import numpy as np
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="SpamShield AI",
    description="Email Spam and Ham Detection using GRU Deep Learning",
    version="1.0.0"
)


# ============================================================
# CORS CONFIGURATION
# ============================================================
#
# Local frontend:
# http://127.0.0.1:5500
# http://localhost:5500
#
# Deployed frontend:
# https://email-spam-detections-427w.onrender.com
#
# IMPORTANT:
# Make sure your deployed frontend URL exactly matches
# the URL shown in your browser.
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://email-spam-detections-427w.onrender.com"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "gru_model.keras"
)

TOKENIZER_PATH = os.path.join(
    BASE_DIR,
    "tokenizer.pickle"
)

CONFIG_PATH = os.path.join(
    BASE_DIR,
    "config.pkl"
)

LABEL_MAPPING_PATH = os.path.join(
    BASE_DIR,
    "label_mapping.pkl"
)


# ============================================================
# GLOBAL VARIABLES
# ============================================================

model_gru = None

tokenizer = None

config = None

label_mapping = None

Stops = set()


# ============================================================
# LOAD NLTK DATA
# ============================================================
#
# We try to download the required resources if they are
# missing.
#
# IMPORTANT:
# On Render, the resources are downloaded during startup.
# For a production setup, you can also install them during
# the build process.
# ============================================================

def setup_nltk():

    global Stops

    try:

        # --------------------------------------------
        # Check punkt
        # --------------------------------------------

        try:

            nltk.data.find(
                "tokenizers/punkt"
            )

        except LookupError:

            print(
                "⚠️ punkt not found. Downloading..."
            )

            nltk.download(
                "punkt",
                quiet=True
            )


        # --------------------------------------------
        # Check punkt_tab
        # --------------------------------------------

        try:

            nltk.data.find(
                "tokenizers/punkt_tab"
            )

        except LookupError:

            print(
                "⚠️ punkt_tab not found. Downloading..."
            )

            nltk.download(
                "punkt_tab",
                quiet=True
            )


        # --------------------------------------------
        # Check stopwords
        # --------------------------------------------

        try:

            nltk.data.find(
                "corpora/stopwords"
            )

        except LookupError:

            print(
                "⚠️ stopwords not found. Downloading..."
            )

            nltk.download(
                "stopwords",
                quiet=True
            )


        # --------------------------------------------
        # Load stopwords
        # --------------------------------------------

        Stops = set(
            stopwords.words(
                "english"
            )
        )

        print(
            "✅ NLTK resources loaded successfully."
        )

        print(
            f"✅ Stopwords loaded: {len(Stops)}"
        )

    except Exception as e:

        print(
            f"❌ NLTK setup error: {e}"
        )

        Stops = set()


# ============================================================
# LOAD GRU MODEL
# ============================================================

def load_model():

    global model_gru

    try:

        print(
            "⏳ Loading GRU model..."
        )

        model_gru = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        print(
            "✅ GRU model loaded successfully."
        )

    except Exception as e:

        print(
            f"❌ Error loading GRU model: {e}"
        )

        model_gru = None


# ============================================================
# LOAD TOKENIZER
# ============================================================

def load_tokenizer():

    global tokenizer

    try:

        print(
            "⏳ Loading tokenizer..."
        )

        with open(
            TOKENIZER_PATH,
            "rb"
        ) as f:

            tokenizer = pickle.load(
                f
            )

        print(
            "✅ Tokenizer loaded successfully."
        )

    except Exception as e:

        print(
            f"❌ Error loading tokenizer: {e}"
        )

        tokenizer = None


# ============================================================
# LOAD CONFIG
# ============================================================

def load_config():

    global config

    try:

        print(
            "⏳ Loading config..."
        )

        with open(
            CONFIG_PATH,
            "rb"
        ) as f:

            config = pickle.load(
                f
            )

        print(
            "✅ Config loaded successfully."
        )

        print(
            f"✅ Config: {config}"
        )

    except Exception as e:

        print(
            f"❌ Error loading config: {e}"
        )

        config = None


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

def load_label_mapping():

    global label_mapping

    try:

        print(
            "⏳ Loading label mapping..."
        )

        with open(
            LABEL_MAPPING_PATH,
            "rb"
        ) as f:

            label_mapping = pickle.load(
                f
            )

        print(
            "✅ Label mapping loaded successfully."
        )

        print(
            f"✅ Label mapping: {label_mapping}"
        )

    except Exception as e:

        print(
            f"❌ Error loading label mapping: {e}"
        )

        label_mapping = None


# ============================================================
# STARTUP EVENT
# ============================================================
#
# All heavy resources are loaded once when FastAPI starts.
#
# This prevents loading the GRU model for every prediction.
# ============================================================

@app.on_event("startup")
def startup_event():

    print(
        ""
    )

    print(
        "============================================================"
    )

    print(
        "🚀 Starting SpamShield AI..."
    )

    print(
        "============================================================"
    )


    # --------------------------------------------
    # NLTK
    # --------------------------------------------

    setup_nltk()


    # --------------------------------------------
    # Model
    # --------------------------------------------

    load_model()


    # --------------------------------------------
    # Tokenizer
    # --------------------------------------------

    load_tokenizer()


    # --------------------------------------------
    # Config
    # --------------------------------------------

    load_config()


    # --------------------------------------------
    # Label mapping
    # --------------------------------------------

    load_label_mapping()


    print(
        "============================================================"
    )

    print(
        "✅ SpamShield AI startup completed."
    )

    print(
        "============================================================"
    )

    print(
        ""
    )


# ============================================================
# TEXT PREPROCESSING
# ============================================================
#
# This follows your original preprocessing:
#
# 1. Lowercase
# 2. Remove punctuation
# 3. Word tokenize
# 4. Remove stopwords
# 5. Join words
# 6. Remove hyperlinks
#
# IMPORTANT:
# Keep preprocessing consistent with your training notebook.
# ============================================================

def preprocess_text(
    text: str
):

    # --------------------------------------------
    # LOWERCASE
    # --------------------------------------------

    text = text.lower()


    # --------------------------------------------
    # REMOVE PUNCTUATION
    # --------------------------------------------

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )


    # --------------------------------------------
    # TOKENIZE
    # --------------------------------------------

    words = word_tokenize(
        text
    )


    # --------------------------------------------
    # REMOVE STOPWORDS
    # --------------------------------------------

    words = [

        word

        for word in words

        if word not in Stops

    ]


    # --------------------------------------------
    # JOIN WORDS
    # --------------------------------------------

    text = " ".join(
        words
    )


    # --------------------------------------------
    # REMOVE HYPERLINKS
    # --------------------------------------------

    text = re.sub(
        r"http\S+",
        "",
        text
    )


    return text


# ============================================================
# REQUEST MODEL
# ============================================================

class EmailRequest(
    BaseModel
):

    email: str


# ============================================================
# RESPONSE MODEL
# ============================================================

class EmailResponse(
    BaseModel
):

    email: str

    prediction: str

    confidence: float

    probability_spam: float

    probability_ham: float


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {

        "message":
            "SpamShield AI API is running",

        "status":
            "success",

        "model":
            "GRU",

        "task":
            "Email Spam/Ham Classification"

    }


# ============================================================
# HEAD ROOT ENDPOINT
# ============================================================

@app.head("/")
def head_home():

    return


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {

        "status":
            "healthy",

        "model_loaded":
            model_gru is not None,

        "tokenizer_loaded":
            tokenizer is not None,

        "config_loaded":
            config is not None,

        "label_mapping_loaded":
            label_mapping is not None,

        "nltk_stopwords_loaded":
            len(Stops) > 0

    }


# ============================================================
# WARMUP ENDPOINT
# ============================================================
#
# This endpoint can be called after the Render service starts
# to initialize the model for inference.
#
# It performs one tiny dummy prediction.
# ============================================================

@app.get("/warmup")
def warmup():

    if model_gru is None:

        raise HTTPException(

            status_code=500,

            detail=
                "GRU model is not loaded."

        )


    if tokenizer is None:

        raise HTTPException(

            status_code=500,

            detail=
                "Tokenizer is not loaded."

        )


    if config is None:

        raise HTTPException(

            status_code=500,

            detail=
                "Config is not loaded."

        )


    try:

        start_time = time.perf_counter()


        # --------------------------------------------
        # Dummy text
        # --------------------------------------------

        dummy_text = "hello"


        # --------------------------------------------
        # Tokenize
        # --------------------------------------------

        sequence = tokenizer.texts_to_sequences(

            [dummy_text]

        )


        # --------------------------------------------
        # Get max length
        # --------------------------------------------

        max_length = config[
            "max_length"
        ]


        # --------------------------------------------
        # Pad
        # --------------------------------------------

        padded_sequence = pad_sequences(

            sequence,

            maxlen=max_length,

            padding="post"

        )


        # --------------------------------------------
        # Run model
        # --------------------------------------------

        model_gru(

            padded_sequence,

            training=False

        ).numpy()


        # --------------------------------------------
        # Calculate time
        # --------------------------------------------

        elapsed_time = (

            time.perf_counter()

            - start_time

        )


        return {

            "status":
                "warm",

            "message":
                "GRU model is ready for prediction.",

            "inference_time_seconds":
                round(
                    elapsed_time,
                    4
                )

        }


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=
                f"Warmup failed: {str(e)}"

        )


# ============================================================
# PREDICT EMAIL
# ============================================================

@app.post(
    "/predict",
    response_model=EmailResponse
)
def predict_email(
    request: EmailRequest
):

    # ========================================================
    # START TIMER
    # ========================================================

    total_start_time = (
        time.perf_counter()
    )


    # ========================================================
    # CHECK MODEL
    # ========================================================

    if model_gru is None:

        raise HTTPException(

            status_code=500,

            detail=
                "GRU model is not loaded."

        )


    # ========================================================
    # CHECK TOKENIZER
    # ========================================================

    if tokenizer is None:

        raise HTTPException(

            status_code=500,

            detail=
                "Tokenizer is not loaded."

        )


    # ========================================================
    # CHECK CONFIG
    # ========================================================

    if config is None:

        raise HTTPException(

            status_code=500,

            detail=
                "Config file is not loaded."

        )


    # ========================================================
    # CHECK LABEL MAPPING
    # ========================================================

    if label_mapping is None:

        raise HTTPException(

            status_code=500,

            detail=
                "Label mapping is not loaded."

        )


    # ========================================================
    # VALIDATE EMAIL
    # ========================================================

    email_text = request.email.strip()


    if not email_text:

        raise HTTPException(

            status_code=400,

            detail=
                "Email text cannot be empty."

        )


    try:

        # ====================================================
        # PREPROCESS EMAIL
        # ====================================================

        preprocess_start_time = (
            time.perf_counter()
        )


        cleaned_text = preprocess_text(
            email_text
        )


        preprocess_time = (

            time.perf_counter()

            - preprocess_start_time

        )


        # ====================================================
        # TOKENIZE
        # ====================================================

        tokenization_start_time = (
            time.perf_counter()
        )


        sequence = tokenizer.texts_to_sequences(

            [cleaned_text]

        )


        # ====================================================
        # GET MAX LENGTH
        # ====================================================

        max_length = config[
            "max_length"
        ]


        # ====================================================
        # PAD SEQUENCE
        # ====================================================

        padded_sequence = pad_sequences(

            sequence,

            maxlen=max_length,

            padding="post"

        )


        tokenization_time = (

            time.perf_counter()

            - tokenization_start_time

        )


        # ====================================================
        # MODEL PREDICTION
        # ====================================================
        #
        # Direct model call is used instead of model.predict()
        # for lower overhead during single-email inference.
        # ====================================================

        prediction_start_time = (
            time.perf_counter()
        )


        prediction = model_gru(

            padded_sequence,

            training=False

        ).numpy()


        prediction_time = (

            time.perf_counter()

            - prediction_start_time

        )


        # ====================================================
        # GET SPAM PROBABILITY
        # ====================================================

        probability_spam = float(

            prediction[0][0]

        )


        # ====================================================
        # PROTECT AGAINST INVALID PROBABILITY
        # ====================================================

        probability_spam = max(

            0.0,

            min(
                1.0,
                probability_spam
            )

        )


        # ====================================================
        # HAM PROBABILITY
        # ====================================================

        probability_ham = (

            1.0

            - probability_spam

        )


        # ====================================================
        # CLASSIFICATION
        # ====================================================

        if probability_spam >= 0.5:

            predicted_label = (

                label_mapping[1]

            )

            confidence = (

                probability_spam

            )

        else:

            predicted_label = (

                label_mapping[0]

            )

            confidence = (

                probability_ham

            )


        # ====================================================
        # TOTAL TIME
        # ====================================================

        total_time = (

            time.perf_counter()

            - total_start_time

        )


        # ====================================================
        # PERFORMANCE LOG
        # ====================================================

        print(
            ""
        )

        print(
            "---------------- PREDICTION PERFORMANCE ----------------"
        )

        print(
            f"Preprocessing time : "
            f"{preprocess_time:.4f} seconds"
        )

        print(
            f"Tokenization time  : "
            f"{tokenization_time:.4f} seconds"
        )

        print(
            f"Model inference    : "
            f"{prediction_time:.4f} seconds"
        )

        print(
            f"Total prediction   : "
            f"{total_time:.4f} seconds"
        )

        print(
            f"Prediction         : "
            f"{predicted_label}"
        )

        print(
            "----------------------------------------------------------"
        )

        print(
            ""
        )


        # ====================================================
        # RETURN RESPONSE
        # ====================================================

        return EmailResponse(

            email=email_text,

            prediction=predicted_label,

            confidence=round(

                confidence,

                4

            ),

            probability_spam=round(

                probability_spam,

                4

            ),

            probability_ham=round(

                probability_ham,

                4

            )

        )


    except Exception as e:

        print(
            f"❌ Prediction error: {e}"
        )


        raise HTTPException(

            status_code=500,

            detail=
                f"Prediction failed: {str(e)}"

        )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "main:app",

        host="0.0.0.0",

        port=int(
            os.environ.get(
                "PORT",
                8000
            )
        ),

        reload=False

    )