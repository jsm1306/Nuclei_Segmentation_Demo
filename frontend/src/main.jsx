import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { predictImage } from "./api";
import "./styles.css";

const MAX_FILE_SIZE = 20 * 1024 * 1024;
const ACCEPTED_TYPES = ["image/png", "image/jpeg"];

function UploadIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" className="upload-icon">
      <path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 14v4a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-4" />
    </svg>
  );
}

function App() {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [results, setResults] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function selectFile(nextFile) {
    setError("");
    setResults(null);
    if (!nextFile) return;
    if (!ACCEPTED_TYPES.includes(nextFile.type)) {
      setError("Choose a PNG or JPG/JPEG microscopy image.");
      return;
    }
    if (nextFile.size > MAX_FILE_SIZE) {
      setError("The image must be smaller than 20 MB.");
      return;
    }
    const nextPreview = URL.createObjectURL(nextFile);
    setFile(nextFile);
    setPreviewUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous);
      return nextPreview;
    });
  }

  function removeFile() {
    setFile(null);
    setResults(null);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
    setPreviewUrl((previous) => {
      if (previous) URL.revokeObjectURL(previous);
      return "";
    });
  }

  async function runPrediction() {
    if (!file || isLoading) return;
    setError("");
    setIsLoading(true);
    try {
      setResults(await predictImage(file));
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsLoading(false);
    }
  }

  function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    selectFile(event.dataTransfer.files[0]);
  }

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="/" aria-label="NuRAFormer home">
          <span className="brand-mark">N</span>
          <span>NuRAFormer</span>
        </a>
        <span className="header-note">Research demonstration</span>
      </header>

      <main>
        <section className="intro-section">
          <div className="intro-copy">
            <p className="eyebrow">Computational microscopy</p>
            <h1>See nuclei boundaries with greater clarity.</h1>
            <p className="intro-text">
              A live demonstration of NuRAFormer, a hybrid UNet–Transformer
              model for nuclei segmentation and boundary-aware analysis.
            </p>
          </div>
          <div className="model-note" aria-label="Model details">
            <span className="model-note-dot" />
            <div>
              <strong>Two-head inference</strong>
              <span>Segmentation + boundary maps</span>
            </div>
          </div>
        </section>

        <section className="workspace-section" aria-labelledby="workspace-title">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Image testing</p>
              <h2 id="workspace-title">Upload a microscopy image</h2>
            </div>
            <p className="section-support">PNG or JPG · resized to 512 × 512 for inference</p>
          </div>

          <div
            className={`upload-zone ${isDragging ? "is-dragging" : ""} ${file ? "has-file" : ""}`}
            onDragEnter={(event) => {
              event.preventDefault();
              setIsDragging(true);
            }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
          >
            {!file ? (
              <>
                <span className="upload-icon-wrap"><UploadIcon /></span>
                <h3>Drop an image here</h3>
                <p>or browse from your computer</p>
                <button className="button button-secondary" onClick={() => inputRef.current?.click()}>
                  Browse files
                </button>
                <input
                  ref={inputRef}
                  className="visually-hidden"
                  type="file"
                  accept=".png,.jpg,.jpeg,image/png,image/jpeg"
                  onChange={(event) => selectFile(event.target.files[0])}
                />
              </>
            ) : (
              <div className="selected-file">
                <div className="selected-preview">
                  <img src={previewUrl} alt="Selected microscopy image preview" />
                </div>
                <div className="selected-details">
                  <p className="file-kicker">Selected image</p>
                  <h3>{file.name}</h3>
                  <p>{(file.size / 1024).toFixed(1)} KB · Ready for prediction</p>
                  <button className="text-button" onClick={removeFile}>Remove image</button>
                </div>
              </div>
            )}
          </div>

          {error && <div className="message message-error" role="alert">{error}</div>}

          <div className="action-row">
            <button className="button button-primary" disabled={!file || isLoading} onClick={runPrediction}>
              {isLoading ? <><span className="spinner" /> Running inference…</> : "Run prediction"}
            </button>
            {file && !isLoading && <button className="text-button reset-button" onClick={removeFile}>Reset</button>}
          </div>
        </section>

        <section className="results-section" aria-labelledby="results-title">
          <div className="section-heading results-heading">
            <div>
              <p className="eyebrow">Model output</p>
              <h2 id="results-title">Prediction results</h2>
            </div>
            {!results && <p className="section-support">Your masks will appear here after inference.</p>}
            {results && <p className="result-status"><span /> Generated at 0.5 threshold</p>}
          </div>

          {!results ? (
            <div className="empty-results">
              <div className="empty-line" />
              <p>Upload an image to inspect its segmentation and boundary maps.</p>
            </div>
          ) : (
            <div className="result-grid">
              <ResultCard label="Processed input · 512 × 512" src={results.original_image} />
              <ResultCard label="Nuclei segmentation mask" src={results.segmentation_mask} />
              <ResultCard label="Nuclei boundary mask" src={results.boundary_mask} />
            </div>
          )}
        </section>
      </main>

      <footer className="site-footer">
        <span>NuRAFormer</span>
        <span>Hybrid UNet–Transformer research project</span>
      </footer>
    </div>
  );
}

function ResultCard({ label, src }) {
  return (
    <figure className="result-card">
      <div className="result-image-wrap">
        <img src={src} alt={label} />
      </div>
      <figcaption>{label}</figcaption>
    </figure>
  );
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode><App /></React.StrictMode>,
);
