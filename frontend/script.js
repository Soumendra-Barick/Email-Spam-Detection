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

  resultContainer.classList.remove("show", "spam", "ham");

  errorBox.classList.remove("show");

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
  const prediction = data.prediction;

  const confidence = data.confidence;

  const spam = data.probability_spam;

  const ham = data.probability_ham;

  // ---------------------------------------------
  // RESET RESULT
  // ---------------------------------------------

  resultContainer.classList.remove("spam", "ham");

  // ---------------------------------------------
  // CHECK PREDICTION
  // ---------------------------------------------

  if (prediction.toLowerCase() === "spam") {
    resultContainer.classList.add("spam");

    resultIcon.textContent = "🚨";

    predictionText.textContent = "Spam Detected";

    resultMessage.textContent =
      "⚠️ This email has been classified as spam. Be careful with links, attachments, and requests for personal information.";
  } else {
    resultContainer.classList.add("ham");

    resultIcon.textContent = "✅";

    predictionText.textContent = "Safe Email";

    resultMessage.textContent =
      "✅ This email has been classified as legitimate (Ham).";
  }

  // ---------------------------------------------
  // CONFIDENCE
  // ---------------------------------------------

  confidenceText.textContent = formatPercentage(confidence);

  confidenceValue.textContent = formatPercentage(confidence);

  // ---------------------------------------------
  // PROBABILITIES
  // ---------------------------------------------

  spamProbability.textContent = formatPercentage(spam);

  hamProbability.textContent = formatPercentage(ham);

  // ---------------------------------------------
  // SHOW RESULT
  // ---------------------------------------------

  resultContainer.classList.add("show");

  // ---------------------------------------------
  // ANIMATE BARS
  // ---------------------------------------------

  setTimeout(function () {
    confidenceBar.style.width = confidence * 100 + "%";

    spamBar.style.width = spam * 100 + "%";

    hamBar.style.width = ham * 100 + "%";
  }, 100);
}

// =========================================================
// ANALYZE EMAIL
// =========================================================

analyzeBtn.addEventListener("click", async function () {
  // ---------------------------------------------
  // GET EMAIL
  // ---------------------------------------------

  const email = emailText.value.trim();

  // ---------------------------------------------
  // VALIDATE
  // ---------------------------------------------

  if (!email) {
    showError("Please enter an email message before analyzing.");

    emailText.focus();

    return;
  }

  // ---------------------------------------------
  // HIDE OLD RESULTS
  // ---------------------------------------------

  hideError();

  resultContainer.classList.remove("show");

  // ---------------------------------------------
  // LOADING
  // ---------------------------------------------

  analyzeBtn.classList.add("loading");

  try {
    // -----------------------------------------
    // API REQUEST
    // -----------------------------------------

    const response = await fetch(API_URL, {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify({
        email: email,
      }),
    });

    // -----------------------------------------
    // HANDLE ERROR
    // -----------------------------------------

    if (!response.ok) {
      let errorText = "Prediction failed.";

      try {
        const errorData = await response.json();

        errorText = errorData.detail || errorText;
      } catch (error) {
        // Ignore JSON parsing error
      }

      throw new Error(errorText);
    }

    // -----------------------------------------
    // GET RESPONSE
    // -----------------------------------------

    const data = await response.json();

    // -----------------------------------------
    // DISPLAY RESULT
    // -----------------------------------------

    displayResult(data);
  } catch (error) {
    console.error(error);

    showError("Unable to connect to the AI server. " + error.message);
  } finally {
    // -----------------------------------------
    // STOP LOADING
    // -----------------------------------------

    analyzeBtn.classList.remove("loading");
  }
});
