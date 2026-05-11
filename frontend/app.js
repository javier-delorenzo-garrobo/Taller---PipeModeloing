const form = document.querySelector("#predictionForm");
const apiUrlInput = document.querySelector("#apiUrl");
const apiStatus = document.querySelector("#apiStatus");
const refreshStatus = document.querySelector("#refreshStatus");
const loadSample = document.querySelector("#loadSample");
const submitButton = document.querySelector("#submitButton");
const emptyState = document.querySelector("#emptyState");
const resultContent = document.querySelector("#resultContent");
const riskBadge = document.querySelector("#riskBadge");
const modelVersion = document.querySelector("#modelVersion");
const grafanaLink = document.querySelector("#grafanaLink");

const fields = [
  "age",
  "sex",
  "chest",
  "resting_blood_pressure",
  "serum_cholestoral",
  "fasting_blood_sugar",
  "resting_electrocardiographic_results",
  "maximum_heart_rate_achieved",
  "exercise_induced_angina",
  "oldpeak",
  "slope",
  "number_of_major_vessels",
  "thal",
];

const samplePatients = [
  {
    age: 54,
    sex: 1,
    chest: 4,
    resting_blood_pressure: 130,
    serum_cholestoral: 246,
    fasting_blood_sugar: 0,
    resting_electrocardiographic_results: 2,
    maximum_heart_rate_achieved: 110,
    exercise_induced_angina: 1,
    oldpeak: 2.1,
    slope: 2,
    number_of_major_vessels: 1,
    thal: 7,
  },
  {
    age: 41,
    sex: 0,
    chest: 2,
    resting_blood_pressure: 110,
    serum_cholestoral: 235,
    fasting_blood_sugar: 0,
    resting_electrocardiographic_results: 0,
    maximum_heart_rate_achieved: 153,
    exercise_induced_angina: 0,
    oldpeak: 0.0,
    slope: 1,
    number_of_major_vessels: 0,
    thal: 3,
  },
];

function apiBaseUrl() {
  return apiUrlInput.value.replace(/\/+$/, "");
}

function setStatus(text, className) {
  apiStatus.textContent = text;
  apiStatus.className = `status-pill ${className}`;
}

function asPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function payloadFromForm() {
  const data = {};
  for (const field of fields) {
    const value = form.elements[field].value;
    data[field] = field === "oldpeak" ? Number.parseFloat(value) : Number.parseInt(value, 10);
  }
  return data;
}

function fillForm(patient) {
  for (const [key, value] of Object.entries(patient)) {
    if (form.elements[key]) {
      form.elements[key].value = value;
    }
  }
}

function renderResult(result) {
  emptyState.classList.add("hidden");
  resultContent.classList.remove("hidden");

  const riskClass = `risk-${result.risk_level.toLowerCase()}`;
  riskBadge.className = `risk-badge ${riskClass}`;
  riskBadge.textContent = `Riesgo ${result.risk_level}`;

  document.querySelector("#predictionLabel").textContent = result.prediction_label;
  document.querySelector("#confidence").textContent = asPercent(result.confidence);
  document.querySelector("#noDiseaseMeter").value = result.probabilities.no_disease;
  document.querySelector("#diseaseMeter").value = result.probabilities.disease;
  document.querySelector("#noDiseaseProb").textContent = asPercent(result.probabilities.no_disease);
  document.querySelector("#diseaseProb").textContent = asPercent(result.probabilities.disease);
  document.querySelector("#inferenceTime").textContent = `${result.inference_time_ms} ms`;
  document.querySelector("#resultVersion").textContent = result.model_version;
  modelVersion.textContent = `Modelo ${result.model_version}`;
  modelVersion.className = "status-pill status-ok";
}

async function checkStatus() {
  setStatus("Comprobando", "status-muted");
  try {
    const response = await fetch(`${apiBaseUrl()}/health`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const health = await response.json();
    setStatus("API activa", "status-ok");
    modelVersion.textContent = `Modelo ${health.model_version}`;
    modelVersion.className = "status-pill status-ok";
  } catch (error) {
    setStatus("API sin conexion", "status-error");
    modelVersion.textContent = "Modelo";
    modelVersion.className = "status-pill status-muted";
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitButton.disabled = true;
  submitButton.textContent = "Prediciendo";

  try {
    const response = await fetch(`${apiBaseUrl()}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payloadFromForm()),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      throw new Error(error.detail || `HTTP ${response.status}`);
    }

    const result = await response.json();
    renderResult(result);
    setStatus("API activa", "status-ok");
  } catch (error) {
    setStatus(error.message, "status-error");
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Predecir";
  }
});

loadSample.addEventListener("click", () => {
  const sample = samplePatients[Math.floor(Math.random() * samplePatients.length)];
  fillForm(sample);
});

refreshStatus.addEventListener("click", checkStatus);

grafanaLink.href = "http://localhost:3000/d/heart-api/heart-disease-model-metrics?orgId=1&refresh=5s";
checkStatus();
