const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

export async function predictImage(file) {
  const formData = new FormData();
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 180000);
  formData.append("file", file);

  let response;
  try {
    response = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
  } catch {
    if (controller.signal.aborted) {
      throw new Error(
        "Inference timed out after 3 minutes. The backend is running on CPU; check its console for the model timing.",
      );
    }
    throw new Error(
      "The backend could not be reached. Start the FastAPI server and try again.",
    );
  } finally {
    window.clearTimeout(timeout);
  }

  let body;
  try {
    body = await response.json();
  } catch {
    throw new Error(`The backend returned an invalid response (${response.status}).`);
  }
  if (!response.ok) {
    throw new Error(body.detail || `Prediction failed (${response.status}).`);
  }
  return body;
}
