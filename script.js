const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const analyzeBtn = document.getElementById("analyzeBtn");
const resultDiv = document.getElementById("result");
const probsDiv = document.getElementById("probabilities");

// show image preview when user selects a file
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) {
    preview.style.display = "none";
    return;
  }
  const url = URL.createObjectURL(file);
  preview.src = url;
  preview.style.display = "block";
});

analyzeBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    alert("Please choose an image first.");
    return;
  }

  resultDiv.textContent = "Analyzing...";
  probsDiv.textContent = "";

  const formData = new FormData();
  // field name must match 'file' in FastAPI endpoint
  formData.append("file", file);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      resultDiv.textContent = "Error: " + response.status;
      return;
    }

    const data = await response.json();
    const prediction = data.prediction;
    const confidence = (data.confidence * 100).toFixed(1);

    resultDiv.textContent = `Prediction: ${prediction} (confidence: ${confidence}%)`;

    if (data.probabilities) {
      const entries = Object.entries(data.probabilities);
      const lines = entries.map(
        ([label, prob]) => `${label}: ${(prob * 100).toFixed(1)}%`
      );
      probsDiv.textContent = "All classes: " + lines.join(" | ");
    }
  } catch (err) {
    console.error(err);
    resultDiv.textContent = "Error calling API.";
  }
});
