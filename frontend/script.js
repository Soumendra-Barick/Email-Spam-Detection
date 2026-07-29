// =========================================================
// FASTAPI CONFIGURATION
// =========================================================

const API_URL = "https://email-spam-detection-eac8.onrender.com";

// =========================================================
// GET ELEMENTS
// =========================================================

const emailText = document.getElementById("emailText");

const charCount = document.getElementById("charCount");

const analyzeBtn = document.getElementById("analyzeBtn");

const clearBtn = document.getElementById("clearBtn");

const resultContainer = document.getElementById("resultContainer");

const errorBox = document.getElementById("errorBox");

const errorMessage = document.getElementById("errorMessage");

const predictionText = document.getElementById("predictionText");

const resultIcon = document.getElementById("resultIcon");

const confidenceText = document.getElementById("confidenceText");

const confidenceValue = document.getElementById("confidenceValue");

const confidenceBar = document.getElementById("confidenceBar");

const spamProbability = document.getElementById("spamProbability");

const hamProbability = document.getElementById("hamProbability");

const spamBar = document.getElementById("spamBar");

const hamBar = document.getElementById("hamBar");

const resultMessage = document.getElementById("resultMessage");

// =========================================================
// CHARACTER COUNTER
// =========================================================

emailText.addEventListener("input", function () {
  const length = emailText.value.length;

  charCount.textContent = `${length.toLocaleString()} characters`;
});

// =========================================================
// CLEAR BUTTON
// =========================================================

clearBtn.addEventListener("click", function () {
  emailText.value = "";

  charCount.textContent = "0 characters";

  // Hide result

  resultContainer.classList.remove("show", "spam", "ham");

  // Hide error

  errorBox.classList.remove("show");

  // Reset bars

  confidenceBar.style.width = "0%";

  spamBar.style.width = "0%";

  hamBar.style.width = "0%";

  // Focus textarea

  emailText.focus();
});

// =========================================================
// SHOW ERROR
// =========================================================

function showError(message) {
  errorMessage.textContent = message;

  errorBox.classList.add("show");
}

// =========================================================
// HIDE ERROR
// =========================================================

function hideError() {
  errorBox.classList.remove("show");
}

// =========================================================
// FORMAT PERCENTAGE
// =========================================================

function formatPercentage(value) {
  return (value * 100).toFixed(2) + "%";
}

// =========================================================
// DISPLAY RESULT
// =========================================================

function displayResult(data) {
  // ---------------------------------------------
  // GET API DATA
  // ---------------------------------------------

  const prediction = data.prediction;

  const confidence = Number(data.confidence);

  const spam = Number(data.probability_spam);

  const ham = Number(data.probability_ham);

  // ---------------------------------------------
  // RESET RESULT CLASSES
  // ---------------------------------------------

  resultContainer.classList.remove("spam", "ham");

  // ---------------------------------------------
  // CHECK SPAM OR HAM
  // ---------------------------------------------

  if (prediction && prediction.toLowerCase() === "spam") {
    // Spam class

    resultContainer.classList.add("spam");

    // Spam icon

    resultIcon.textContent = "🚨";

    // Spam title

    predictionText.textContent = "Spam Detected";

    // Spam message

    resultMessage.textContent =
      "⚠️ This email has been classified as spam. Be careful with links, attachments, and requests for personal information.";
  } else {
    // Ham class

    resultContainer.classList.add("ham");

    // Ham icon

    resultIcon.textContent = "✅";

    // Ham title

    predictionText.textContent = "Safe Email";

    // Ham message

    resultMessage.textContent =
      "✅ This email has been classified as legitimate (Ham).";
  }

  // ---------------------------------------------
  // CONFIDENCE
  // ---------------------------------------------

  confidenceText.textContent = formatPercentage(confidence);

  confidenceValue.textContent = formatPercentage(confidence);

  // ---------------------------------------------
  // SPAM PROBABILITY
  // ---------------------------------------------

  spamProbability.textContent = formatPercentage(spam);

  // ---------------------------------------------
  // HAM PROBABILITY
  // ---------------------------------------------

  hamProbability.textContent = formatPercentage(ham);

  // ---------------------------------------------
  // SHOW RESULT
  // ---------------------------------------------

  resultContainer.classList.add("show");

  // ---------------------------------------------
  // ANIMATE PROGRESS BARS
  // ---------------------------------------------

  setTimeout(function () {
    confidenceBar.style.width = `${confidence * 100}%`;

    spamBar.style.width = `${spam * 100}%`;

    hamBar.style.width = `${ham * 100}%`;
  }, 100);
}

// =========================================================
// ANALYZE EMAIL
// =========================================================

analyzeBtn.addEventListener("click", async function () {
  // ---------------------------------------------
  // GET EMAIL TEXT
  // ---------------------------------------------

  const email = emailText.value.trim();

  // ---------------------------------------------
  // VALIDATE EMAIL
  // ---------------------------------------------

  if (!email) {
    showError("Please enter an email message before analyzing.");

    emailText.focus();

    return;
  }

  // ---------------------------------------------
  // HIDE PREVIOUS ERROR
  // ---------------------------------------------

  hideError();

  // ---------------------------------------------
  // HIDE PREVIOUS RESULT
  // ---------------------------------------------

  resultContainer.classList.remove("show");

  // ---------------------------------------------
  // RESET PROGRESS BARS
  // ---------------------------------------------

  confidenceBar.style.width = "0%";

  spamBar.style.width = "0%";

  hamBar.style.width = "0%";

  // ---------------------------------------------
  // START LOADING
  // ---------------------------------------------

  analyzeBtn.classList.add("loading");

  try {
    // =================================================
    // IMPORTANT:
    // FastAPI endpoint is:
    //
    // @app.post("/predict")
    //
    // Therefore we MUST call:
    //
    // /predict
    // =================================================

    const response = await fetch(`${API_URL}/predict`, {
      // MUST be POST

      method: "POST",

      // JSON header

      headers: {
        "Content-Type": "application/json",
      },

      // FastAPI expects:
      //
      // {
      //   "email": "your email text"
      // }

      body: JSON.stringify({
        email: email,
      }),
    });

    // ---------------------------------------------
    // CHECK RESPONSE STATUS
    // ---------------------------------------------

    console.log("API Status:", response.status);

    // ---------------------------------------------
    // HANDLE API ERROR
    // ---------------------------------------------

    if (!response.ok) {
      let errorText = `Server error: ${response.status}`;

      try {
        const errorData = await response.json();

        if (errorData.detail) {
          errorText = errorData.detail;
        }
      } catch (jsonError) {
        console.error("Error reading server response:", jsonError);
      }

      throw new Error(errorText);
    }

    // ---------------------------------------------
    // GET JSON RESPONSE
    // ---------------------------------------------

    const data = await response.json();

    // ---------------------------------------------
    // DEBUG RESPONSE
    // ---------------------------------------------

    console.log("Prediction response:", data);

    // ---------------------------------------------
    // DISPLAY RESULT
    // ---------------------------------------------

    displayResult(data);
  } catch (error) {
    console.error("Prediction Error:", error);

    showError("Prediction failed: " + error.message);
  } finally {
    // ---------------------------------------------
    // STOP LOADING
    // ---------------------------------------------

    analyzeBtn.classList.remove("loading");
  }
});
