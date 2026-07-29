import os
import pickle
import string
import re

import numpy as np
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="SpamShield AI",
    description="Email Spam and Ham Detection using GRU Deep Learning",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
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
# LOAD GRU MODEL
# ============================================================

try:

    model_gru = tf.keras.models.load_model(
        MODEL_PATH
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

try:

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

try:

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

except Exception as e:

    print(
        f"❌ Error loading config: {e}"
    )

    config = None


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

try:

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

except Exception as e:

    print(
        f"❌ Error loading label mapping: {e}"
    )

    label_mapping = None


# ============================================================
# STOPWORDS
# ============================================================

try:

    Stops = set(
        stopwords.words(
            "english"
        )
    )

except Exception:

    Stops = set()


# ============================================================
# TEXT PREPROCESSING
#
# This exactly follows your notebook:
#
# 1. Lowercase
# 2. Remove punctuation
# 3. Word tokenize
# 4. Remove stopwords
# 5. Join words
# 6. Remove hyperlinks
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
            label_mapping is not None

    }


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

        cleaned_text = preprocess_text(
            email_text
        )


        # ====================================================
        # TOKENIZE
        # ====================================================

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


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction = model_gru.predict(

            padded_sequence,

            verbose=0

        )


        # ====================================================
        # GET SPAM PROBABILITY
        #
        # Your model:
        #
        # Dense(1, activation='sigmoid')
        #
        # Therefore output is probability of class 1.
        #
        # Your mapping:
        #
        # 0 = Ham
        # 1 = Spam
        # ====================================================

        probability_spam = float(

            prediction[0][0]

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

        raise HTTPException(

            status_code=500,

            detail=
            f"Prediction failed: {str(e)}"

        )