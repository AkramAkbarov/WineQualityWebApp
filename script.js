const API_BASE = window.location.origin;

const FEATURES = [
  { key: "fixed_acidity", min: 3, max: 16 },
  { key: "volatile_acidity", min: 0, max: 2 },
  { key: "citric_acid", min: 0, max: 1.5 },
  { key: "residual_sugar", min: 0, max: 70 },
  { key: "chlorides", min: 0, max: 1 },
  { key: "free_sulfur_dioxide", min: 0, max: 350 },
  { key: "total_sulfur_dioxide", min: 0, max: 450 },
  { key: "density", min: 0.98, max: 1.04 },
  { key: "pH", min: 2.5, max: 4.5 },
  { key: "sulphates", min: 0, max: 2.5 },
  { key: "alcohol", min: 5, max: 16 },
];

const form = document.getElementById("wine-form");
const predictBtn = document.getElementById("predict-btn");
const resetBtn = document.getElementById("reset-btn");
const resultPanel = document.getElementById("result-panel");
const errorPanel = document.getElementById("error-panel");
const resultText = document.getElementById("result-text");
const confidenceText = document.getElementById("confidence-text");
const probBars = document.getElementById("prob-bars");
const errorText = document.getElementById("error-text");

function setFieldError(key, message) {
  const input = document.getElementById(key);
  const msgEl = document.querySelector(`[data-for="${key}"]`);
  if (message) {
    input.classList.add("invalid");
    msgEl.textContent = message;
  } else {
    input.classList.remove("invalid");
    msgEl.textContent = "";
  }
}

function clearErrors() {
  FEATURES.forEach(({ key }) => setFieldError(key, ""));
}

function validateForm() {
  clearErrors();
  let valid = true;
  const payload = {};

  for (const { key, min, max } of FEATURES) {
    const input = document.getElementById(key);
    const raw = input.value.trim();

    if (raw === "") {
      setFieldError(key, "Bu sahə mütləqdir.");
      valid = false;
      continue;
    }

    const value = Number(raw);
    if (Number.isNaN(value)) {
      setFieldError(key, "Düzgün rəqəm daxil edin.");
      valid = false;
      continue;
    }

    if (value < min || value > max) {
      setFieldError(key, `Dəyər ${min} – ${max} aralığında olmalıdır.`);
      valid = false;
      continue;
    }

    payload[key] = value;
  }

  const free = payload.free_sulfur_dioxide;
  const total = payload.total_sulfur_dioxide;
  if (
    valid &&
    free !== undefined &&
    total !== undefined &&
    free > total
  ) {
    setFieldError(
      "free_sulfur_dioxide",
      "Free SO₂ total SO₂-dən böyük ola bilməz."
    );
    valid = false;
  }

  return valid ? payload : null;
}

function setLoading(isLoading) {
  predictBtn.disabled = isLoading;
  predictBtn.classList.toggle("loading", isLoading);
}

function hidePanels() {
  resultPanel.classList.add("hidden");
  errorPanel.classList.add("hidden");
}

function showError(message) {
  errorText.textContent = message;
  errorPanel.classList.remove("hidden");
  resultPanel.classList.add("hidden");
}

function renderProbabilities(probabilities, predictedClass, tierNames) {
  probBars.innerHTML = "";
  const entries = Object.entries(probabilities).sort(
    (a, b) => Number(a[0].split("_")[1]) - Number(b[0].split("_")[1])
  );

  entries.forEach(([label, prob]) => {
    const classIdx = Number(label.split("_")[1]);
    const tierLabel =
      Array.isArray(tierNames) && tierNames[classIdx]
        ? tierNames[classIdx]
        : `Sinif ${classIdx}`;
    const row = document.createElement("div");
    row.className =
      "prob-row" + (classIdx === predictedClass ? " highlight" : "");

    row.innerHTML = `
      <span class="prob-label" title="Model indeksi: ${classIdx}">${tierLabel}</span>
      <div class="prob-track">
        <div class="prob-fill" style="width: ${(prob * 100).toFixed(1)}%"></div>
      </div>
      <span class="prob-pct">${(prob * 100).toFixed(1)}%</span>
    `;

    probBars.appendChild(row);
  });
}

function showResult(data) {
  const { predicted_class, predicted_label, confidence, probabilities, tier_names } = data;
  resultText.textContent = `Proqnoz: «${predicted_label}»`;
  confidenceText.textContent = `Etibar: ${(confidence * 100).toFixed(1)}% · model sinfi: ${predicted_class} (0–5)`;
  renderProbabilities(probabilities, predicted_class, tier_names);
  resultPanel.classList.remove("hidden");
  errorPanel.classList.add("hidden");
}

async function predict(payload) {
  setLoading(true);
  hidePanels();

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      const detail = data.detail;
      let message = "Proqnoz alınmadı. Yenidən cəhd edin.";

      if (typeof detail === "string") {
        message = detail;
      } else if (Array.isArray(detail)) {
        message = detail.map((d) => d.msg || JSON.stringify(d)).join(" ");
      }

      showError(message);
      return;
    }

    showResult(data);
  } catch {
    showError(
      "Serverə qoşulmaq mümkün olmadı. FastAPI işləyir? (uvicorn main:app --reload)"
    );
  } finally {
    setLoading(false);
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const payload = validateForm();
  if (payload) {
    predict(payload);
  }
});

resetBtn.addEventListener("click", () => {
  form.reset();
  clearErrors();
  hidePanels();
});

FEATURES.forEach(({ key }) => {
  document.getElementById(key).addEventListener("input", () => {
    setFieldError(key, "");
  });
});

async function checkBackend() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    if (!response.ok) {
      showError("Backend cavab vermir. Serveri yenidən başladın.");
      return;
    }

    const data = await response.json();

    if (!data.model_loaded) {
      const reason = data.error
        ? `Model yüklənməyib: ${data.error}`
        : "Model yüklənməyib. model.pth faylını yoxlayın və serveri yenidən başladın.";
      showError(reason);
      predictBtn.disabled = true;
      return;
    }

    if (!data.scaler_loaded) {
      console.warn(
        "scaler.pkl yoxdur — təlimdə scaler istifadə etmisinizsə, create_scaler.py işə salın."
      );
    }
  } catch {
    showError(
      "Səhifəni birbaşa açmayın (file://). Terminalda işə salın: uvicorn main:app --reload --port 8000 və http://127.0.0.1:8000 açın."
    );
    predictBtn.disabled = true;
  }
}

checkBackend();
