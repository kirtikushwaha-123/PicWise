# Phase 11B — Production Packaging & Deployment Hardening Audit

## 1. Executive Summary

PicWise is an AI-powered product-label analysis system supporting two distinct product domains: **Food** and **Personal Care**. Phase 11A established that the core application is functionally complete, robust, and production-ready, with all 126 regression tests passing (57 Food, 58 Personal Care, 11 Remediation) and 100% compliance across category routing, domain isolation, and failure safety semantics (*Unknown != Safe*, *Missing Data != Safe*, *OCR Failure != Safe*).

This Phase 11B Part 1 audit assesses the deployment, packaging, and operational hardening requirements of the repository. The audit evaluates startup architecture, WSGI server configurations, OCR/OpenMP worker lifecycle and memory behavior, legacy endpoint security, exception exposure, containerization feasibility, environment handling, filesystem state, health check requirements, logging, and dependency reproducibility.

The audit confirms **zero P0 (critical/system-breaking)** and **zero P1 (serious deployment blocker)** issues. One **P2 (meaningful production concern)** exists: CPU OCR inference latency (35–110s on complex packaging) and OpenMP thread pool accumulation under sustained load. Three **P3 (low-risk hardening)** items were identified: lack of pre-read 16MB file size check on legacy `/api/analyze`, raw exception string exposure in 500 JSON responses, and default `debug=True` in `app.py`. A structured Part 2 implementation plan is defined to resolve these items cleanly without altering any frozen ML models, OCR pipelines, or scoring methodologies.

---

## 2. Repository Baseline

The audit was conducted on a clean working tree synchronized with `origin/master`.

### Git Verification
- **Branch:** `master`
- **Active Commit:** `25cb7a9` (`docs: complete Phase 11A production readiness audit`)
- **Remote Synchronization:** Up to date with `origin/master`
- **Working Tree State:** Clean (no modified, staged, or untracked production files)

### Recent Commit History
```text
25cb7a9 docs: complete Phase 11A production readiness audit
1964957 fix: improve personal care OCR robustness
a876a2d test: validate personal care real-world robustness
e920d3a test: complete personal care production validation
c778e23 feat: integrate personal care analysis pipeline
```

---

## 3. Current Startup Architecture

### Application Startup Path
The application entrypoint is defined in [`app.py`](file:///C:/Users/velzyaa/Desktop/PicWise/app.py):
```python
from backend import create_app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
```

### Application Factory
Defined in [`backend/__init__.py`](file:///C:/Users/velzyaa/Desktop/PicWise/backend/__init__.py):
- `create_app()` initializes a Flask instance:
  ```python
  app = Flask(__name__, static_folder="../static", template_folder="../templates")
  ```
- Loads knowledge bases via `KnowledgeBase.from_env()` into `app.config["KNOWLEDGE_BASE"]`.
- Configures payload ceiling: `app.config.setdefault("MAX_CONTENT_LENGTH", 16 * 1024 * 1024)` (16 MB).
- Registers `api_bp` (prefix `/api`).
- Registers HTTP 413 error handler returning structured JSON:
  ```json
  {"error": "Uploaded image file exceeds the maximum allowed size of 16MB.", "success": false}
  ```
- Defines view routes for HTML pages: `/` (home), `/login` (login), and `/upload` (upload).

### Findings
1. **Startup Method:** Currently started directly via `python app.py` (development server).
2. **Development-Only Nature of `app.py`:** The block `if __name__ == "__main__": app.run(debug=True)` is intended solely for local development.
3. **WSGI Compatibility:** The module-level variable `app` in `app.py` is already an instantiated, WSGI-compliant WSGI callable (`app = create_app()`).
4. **Production WSGI Entrypoint:** No dedicated production entrypoint (e.g. `wsgi.py`) or WSGI server configuration (e.g. `gunicorn.conf.py`) currently exists.
5. **Development Server Usage:** Flask's built-in development server is only invoked if `app.py` is executed as `__main__`.
6. **Debug Mode Exposure:** If `python app.py` is executed in a production container, `debug=True` is activated by default.
7. **Configuration:** Core settings (`MAX_CONTENT_LENGTH`, static/template paths) are configured in code; dataset paths are optionally environment-driven with safe built-in fallbacks.

---

## 4. WSGI Deployment Assessment

Running Flask's built-in development server in production is insecure and unperformant. A production WSGI server is required.

### Target Deployment Environments

#### A. Linux / Container Production (Canonical Production Path)
- **Recommended WSGI Server:** `gunicorn`
- **Worker Class:** `sync` (synchronous process workers).
  - *Rationale:* PaddleOCR (PaddlePaddle C++ runtime) and OpenCV use OpenMP and multithreaded native libraries. Threaded workers (`gthread`) or async workers (`gevent`, `eventlet`) cause severe GIL contention, OpenMP thread explosions, and race conditions in native C++ libraries.
- **Worker Count:** 2 to 4 workers.
  - *Formula:* `workers = max(2, min(4, os.cpu_count() or 2))`.
  - *Rationale:* Each OCR extraction on CPU utilizes multiple cores via OpenMP. Running too many concurrent worker processes would cause CPU thrashing and memory exhaustion.
- **Worker Timeout:** 180 seconds.
  - *Rationale:* Complex personal care packaging with dense ingredient lists and low contrast requires 35–110 seconds of CPU inference. The default Gunicorn timeout of 30 seconds will prematurely terminate worker processes mid-analysis.
- **Graceful Timeout:** 30 seconds.
- **Worker Recycling:** `max_requests = 100`, `max_requests_jitter = 20`.
  - *Rationale:* Restarts each worker process after 80–120 requests to release accumulated native OpenMP thread pools and mitigate memory fragmentation.
- **Preload Behavior:** `preload_app = False`.
  - *Rationale:* Preloading before process forking can corrupt C++ runtime handles (PaddlePaddle/PyTorch/OpenCV) across forks. Each worker should initialize its own models cleanly.

#### B. Windows Production-Like / Staging Execution
- Gunicorn does not support Windows (relies on POSIX `fork` and signals).
- **Recommended Windows WSGI Server:** `waitress`
  - Command: `waitress-serve --listen=0.0.0.0:5000 app:app`
  - Compatible with Windows threads and pure Python WSGI specification.

---

## 5. OCR Worker / OpenMP Assessment

### Model Loading & Memory Architecture
- **PaddleOCR Initialization:** In [`backend/services/ocr_service/ocr/paddle_engine.py`](file:///C:/Users/velzyaa/Desktop/PicWise/backend/services/ocr_service/ocr/paddle_engine.py), `_init_engine()` initializes `PaddleOCR` once lazily and caches it in global `_OCR_ENGINE`.
- **ML Model Caching:**
  - Food Safety model is loaded as a module-level cached singleton in `backend/ml/models/food_safety/`.
  - Personal Care models (Safety, Allergy, Irritation pipelines) are loaded as cached singletons in `backend/services/personal_care_analysis_service/models.py`.
- **OpenMP & Multithreading Behavior:**
  - PaddlePaddle utilizes OpenMP (`libgomp` on Linux, `vcomp` on Windows) for tensor computations.
  - Under sustained sequential batch execution, OpenMP thread pools can persist in memory.
  - Without explicit limits, OpenMP defaults to spawning threads equal to the total number of hardware CPU threads. In a multi-worker setup, this leads to heavy core contention.

### Recommended Operational Controls
1. **Thread Limits via Environment Variables:**
   - `OMP_NUM_THREADS=2`
   - `MKL_NUM_THREADS=2`
   - `OPENBLAS_NUM_THREADS=2`
   - `VECLIB_MAXIMUM_THREADS=2`
   - `NUMEXPR_NUM_THREADS=2`
   Setting these in the container environment prevents any single worker from monopolizing all CPU cores during OCR inference.
2. **Process Recycling:**
   - Process recycling (`max_requests` in Gunicorn) completely terminates the OS process, thereby reclaiming all heap memory, native memory allocations, and OpenMP thread pools. Thread recycling is insufficient because native C++ memory is not reclaimed by Python thread termination.

---

## 6. Legacy `/api/analyze` Assessment

### Code Inspection
In [`backend/routes/api.py`](file:///C:/Users/velzyaa/Desktop/PicWise/backend/routes/api.py) (lines 113–145):
```python
@api_bp.post("/analyze")
def analyze():
    category = request.form.get("category")
    if not category or not category.strip():
        return jsonify({"error": "Product category is required in form field 'category'."}), 400

    category = category.strip().lower()
    if category not in ("food", "personal_care"):
        return jsonify({
            "error": f"Invalid category '{category}'. Allowed values are 'food' or 'personal_care'."
        }), 400

    image = request.files.get("image")
    if image is None or image.filename == "":
        return jsonify({"error": "Image file is required in form field 'image'."}), 400

    if not _is_allowed_image(image.mimetype, image.filename):
        return jsonify({"error": "Only JPG, JPEG, PNG, and WEBP images are supported."}), 400

    image_bytes = image.read()
    if not image_bytes or len(image_bytes) == 0:
        return jsonify({"error": "Uploaded image file is empty."}), 400

    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.verify()
    except Exception:
        return jsonify({"error": "Invalid or corrupt image file."}), 400

    knowledge_base = current_app.config.get("KNOWLEDGE_BASE")
    result = analyze_product_image(image_bytes, knowledge_base, category=category)
    return jsonify(result)
```

### Assessment & Gap Analysis
1. **File Size Check:** Unlike `/api/food/analyze` and `/api/personal-care/analyze`, `/api/analyze` does **not** check `if len(image_bytes) > MAX_IMAGE_SIZE_BYTES: return ..., 413`.
   - While Flask's `MAX_CONTENT_LENGTH` will reject requests when `Content-Length` header exceeds 16MB, if chunked transfer or boundary streaming occurs, the endpoint will read arbitrary bytes into RAM.
2. **Exception Handling:** `/api/analyze` lacks a `try...except` block around `analyze_product_image`. An unexpected error results in an unhandled exception or default HTML 500 page.
3. **Usage Search:**
   - **Frontend:** [`static/app.js`](file:///C:/Users/velzyaa/Desktop/PicWise/static/app.js) was audited. It exclusively routes to `/api/food/analyze` and `/api/personal-care/analyze`. It **never** calls `/api/analyze`.
   - **Tests:** `tests/test_food_analysis_pipeline.py`, `tests/test_food_safety_production.py`, `tests/test_analysis.py`, and `tests/test_ocr_integration.py` test `/api/analyze` for backward compatibility.
   - **Documentation:** `README.md` documents `/api/analyze`.
4. **Decision:** **Option A — Keep and Harden.** Retain `/api/analyze` for backward compatibility, but harden it in Part 2 with the identical 16MB pre-read ceiling check and sanitized 500 error handling.

---

## 7. Error Handling & Exception Exposure

### Code Inspection
In [`backend/routes/api.py`](file:///C:/Users/velzyaa/Desktop/PicWise/backend/routes/api.py):
- Lines 55–61 (`/api/food/analyze`):
  ```python
  except Exception as exc:
      current_app.logger.error(f"Unexpected error in /api/food/analyze: {exc}", exc_info=True)
      return jsonify({
          "error": "An unexpected server error occurred while analyzing the food product.",
          "errors": [str(exc)],
          "success": False,
      }), 500
  ```
- Lines 104–110 (`/api/personal-care/analyze`):
  ```python
  except Exception as exc:
      current_app.logger.error(f"Unexpected error in /api/personal-care/analyze: {exc}", exc_info=True)
      return jsonify({
          "error": "An unexpected server error occurred while analyzing the personal care product.",
          "errors": [str(exc)],
          "success": False,
      }), 500
  ```

### Assessment & Exposure Risk
1. **Server Logging:** Both endpoints correctly call `current_app.logger.error(..., exc_info=True)`, recording the full exception message and stack trace in server logs.
2. **Client Leakage:** Returning `"errors": [str(exc)]` directly in the HTTP 500 JSON response exposes internal implementation details (e.g. internal file paths, module names, unhandled database or C-extension error strings) to untrusted clients.
3. **Proposed Part 2 Remediation:**
   - Server-side: Retain detailed logging with `exc_info=True`.
   - Client-side: Return generic, sanitized error messages:
     ```json
     {
       "error": "An unexpected server error occurred while analyzing the product.",
       "errors": ["An unexpected server error occurred. Please try again."],
       "success": false
     }
     ```
   - Apply the same pattern to `/api/analyze`.

---

## 8. Debug Mode Assessment

### Code Inspection
In [`app.py`](file:///C:/Users/velzyaa/Desktop/PicWise/app.py):
```python
if __name__ == "__main__":
    app.run(debug=True)
```

### Assessment
1. **Risk:** If a container or deployment script executes `python app.py`, Flask's interactive Werkzeug debugger is enabled. In production, this allows arbitrary code execution if the debugger pin is obtained or if debug endpoints are accessible.
2. **WSGI Execution:** When running under a WSGI server (e.g. `gunicorn app:app`), `app.py` is imported as a module, so the `if __name__ == "__main__":` block does **not** execute, preventing `debug=True` from being invoked.
3. **Hardening in Part 2:**
   - Update `app.py` to check an environment variable:
     ```python
     debug_mode = os.getenv("FLASK_DEBUG", "false").lower() in ("true", "1")
     app.run(debug=debug_mode)
     ```
   - Provide an explicit `wsgi.py` production entrypoint that cleanly exposes `application = create_app()`.

---

## 9. Containerization Assessment

### Repository Inventory
- **Docker Artifacts:** No `Dockerfile`, `docker-compose.yml`, or `.dockerignore` exists.
- **Backend Architecture:** Pure Python 3.13 application with Flask.
- **Frontend Assets:** Static HTML (`templates/`), CSS, and vanilla JS (`static/`). Zero Node.js, npm, or frontend bundler dependencies.
- **Database / External Services:**
  - Zero SQL databases (SQLite, PostgreSQL, MySQL).
  - Zero NoSQL databases.
  - Zero Neo4j dependencies in application code.
  - All knowledge bases are loaded from local CSV/XLSX into memory at startup.
- **Model Artifacts:**
  - 10 `.joblib` model files stored in `backend/ml/models/`.
  - PaddleOCR models are fetched and cached in `~/.paddlex/official_models/` or `~/.paddleocr/`.

### Containerization Feasibility
PicWise is exceptionally well-suited for containerization. It is fully self-contained and stateless, requiring only a Python base image, system dependencies for OpenCV/OpenMP (e.g. `libgl1`, `libgomp1`), and model weights.

---

## 10. Environment & Secret Handling

### Configuration Loading
- Environment variables are queried in [`backend/services/knowledge_base.py`](file:///C:/Users/velzyaa/Desktop/PicWise/backend/services/knowledge_base.py) and [`backend/services/ocr_service/config.py`](file:///C:/Users/velzyaa/Desktop/PicWise/backend/services/ocr_service/config.py):
  - `FOOD_DATA_PATH`: Path to food ingredients dataset.
  - `PERSONAL_CARE_DATA_PATH`: Path to personal care ingredients dataset.
  - `NUTRITION_DATA_PATH`: Path to nutrition knowledge dataset.
- All three variables have built-in default paths pointing to `data/` within the repository.
- No mandatory environment variables exist. Missing variables result in graceful fallback to repository defaults.
- **Secrets Audit:** Zero secrets, API keys, tokens, or credentials are required or embedded in the codebase.
- **`.env` Handling:** `.env` is listed in [`.gitignore`](file:///C:/Users/velzyaa/Desktop/PicWise/.gitignore). [`.env.example`](file:///C:/Users/velzyaa/Desktop/PicWise/.env.example) documents the optional dataset path variables without exposing any sensitive information.

---

## 11. Filesystem & Runtime State

### Filesystem Write Operations
- An audit of all write operations across `backend/` was conducted.
- **Request Processing:** Neither `/api/food/analyze`, `/api/personal-care/analyze`, nor `/api/analyze` writes any files to the filesystem. Uploaded image bytes are processed entirely in memory via `io.BytesIO` and `cv2.imdecode`.
- **Upload Directory:** `static/uploads/` exists with a `.gitkeep`, but is never written to by the application.
- **Model Files:** Loaded read-only via `joblib.load()`.
- **Cache Directories:** PaddleOCR requires a writable cache directory for downloaded model weights (`~/.paddlex/` or `~/.paddleocr/`).

### Container Filesystem Compatibility
- The application container root filesystem can be mounted **read-only** (`--read-only`), provided that:
  1. `/tmp` is mounted as a writable `tmpfs`.
  2. The user home directory (e.g. `/home/appuser`) is writable or pre-populated with PaddleOCR model weights during the container build.
- Standard logs are emitted to `stdout`/`stderr`, requiring no writable log directory.

---

## 12. Health / Readiness Assessment

### Current State
- No health check endpoint (`/health`, `/healthz`, `/ready`) currently exists in the application.

### Assessment
- In container orchestrators (Docker Swarm, Kubernetes, AWS ECS, GCP Cloud Run), a health endpoint is required to monitor container liveness and readiness.
- **Proposed Part 2 Implementation:**
  - Route: `GET /health` (or `GET /api/health`).
  - Response: HTTP 200 with JSON:
    ```json
    {
      "status": "healthy",
      "app": "PicWise",
      "version": "1.0.0",
      "knowledge_base_loaded": true
    }
    ```
  - *Critical Constraint:* The health check must be lightweight and must **not** trigger OCR inference or heavy ML prediction, preventing CPU spikes and false timeout failures.

---

## 13. Logging & Observability

### Current State
- Uses Flask's standard `current_app.logger` (Python `logging` module).
- All unhandled exceptions in `/api/food/analyze` and `/api/personal-care/analyze` are logged with `exc_info=True`.
- Logs are emitted to standard output/standard error (`stdout`/`stderr`).
- No file-based log handlers or third-party observability dependencies exist.

### Container Observability
- Emitting logs to `stdout`/`stderr` aligns with the 12-factor application methodology.
- Container runtimes (Docker, containerd) automatically capture these streams for aggregation.
- In Part 2, a standardized logging configuration (timestamps, log levels, logger name) should be initialized in `create_app()` or WSGI startup.

---

## 14. Dependency Reproducibility

### Active Environment vs. `requirements.txt`
The active virtual environment (`.venv`) runs Python 3.13.2. A comparison was performed between `requirements.txt` and `pip list`:

| Package | In `requirements.txt` | In `.venv` | Status / Risk |
|---|---|---|---|
| `Flask` | `Flask==3.0.3` | `3.1.3` | Minor version drift |
| `openpyxl` | `openpyxl==3.1.5` | `3.1.5` | Pinned & matched |
| `pandas` | `pandas>=2.0.0` | `3.0.5` | Unpinned upper bound |
| `scikit-learn` | `scikit-learn>=1.3.0` | `1.9.1` | Unpinned upper bound |
| `xgboost` | `xgboost>=2.0.0` | `3.4.1` | Unpinned upper bound |
| `joblib` | `joblib>=1.3.0` | `1.6.0` | Unpinned upper bound |
| `paddleocr` | **MISSING** | `3.7.0` | Required for OCR |
| `paddlepaddle` | **MISSING** | `3.3.1` | Required for PaddleOCR |
| `paddlex` | **MISSING** | `3.7.2` | Required for PaddleOCR 3.x |
| `pillow` | **MISSING** | `12.3.0` | Required for image validation |
| `opencv-python` | **MISSING** | `5.0.0.93` | Required for preprocessing |
| `RapidFuzz` | **MISSING** | `3.14.6` | Required for ingredient matching |
| `numpy` | **MISSING** | `2.3.5` | Required for tensor/array ops |
| `scipy` | **MISSING** | `1.18.1` | Required for scikit-learn |
| `gunicorn` | **MISSING** | **NOT INSTALLED** | Required for Linux WSGI |
| `waitress` | **MISSING** | **NOT INSTALLED** | Required for Windows WSGI |

### Gap Assessment
`requirements.txt` currently specifies only 6 packages and omits critical runtime dependencies (`paddleocr`, `paddlepaddle`, `pillow`, `opencv-python`, `rapidfuzz`). For container builds to be reproducible, a comprehensive `requirements.txt` (or `requirements-prod.txt`) must be established in Part 2.

---

## 15. Static / Template Serving

### Current Architecture
- Flask directly serves static assets from `static/` via `static_folder="../static"`.
- Flask directly renders HTML templates from `templates/` via `template_folder="../templates"`.
- Static files include `style.css`, `app.js`, and vector icons in `static/images/`.

### Production Suitability
- Under Gunicorn or Waitress, Flask serves these assets natively with low overhead.
- No client-side compilation or node build steps are required.
- In high-traffic deployments, a reverse proxy (e.g. Nginx or Cloudflare CDN) can optionally terminate SSL and cache `/static/*`, while passing `/api/*` and view routes to WSGI.

---

## 16. Container Security

A future production container must adhere to the following security standards:
1. **Non-Root Execution:** Create a dedicated unprivileged user (e.g. `appuser`, UID 10001) and execute the WSGI server under this user.
2. **Minimal Base Image:** Use `python:3.13-slim` or a verified Linux base image with minimal system libraries (`libgl1`, `libgomp1`).
3. **Artifact Exclusion (`.dockerignore`):**
   - Exclude `.git`, `.venv`, `__pycache__`, `*.pyc`, `.env`, `*.log`, `scratch/`, and development scripts.
4. **Port Exposure:** Expose only port 5000.
5. **No Embedded Secrets:** Ensure no secrets or API keys are embedded in container layers.
6. **Read-Only Root Filesystem:** Support running with `--read-only` and `tmpfs` mounts for `/tmp` and cache directories.

---

## 17. Part 2 Implementation Plan

The following concrete actions are planned for Phase 11B Part 2:

1. **Production WSGI Configuration:**
   - Create `wsgi.py` exposing `application = create_app()`.
   - Create `gunicorn.conf.py` specifying:
     - `bind = "0.0.0.0:5000"`
     - `workers = 2` (process-based sync workers)
     - `timeout = 180`
     - `graceful_timeout = 30`
     - `max_requests = 100`
     - `max_requests_jitter = 20`
     - `preload_app = False`
   - Provide Windows `waitress` entrypoint/instructions.
2. **OpenMP / Thread Pool Hardening:**
   - Configure container environment variables: `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2`.
3. **Legacy `/api/analyze` Hardening:**
   - Add pre-read 16MB ceiling check: `if len(image_bytes) > MAX_IMAGE_SIZE_BYTES: return ..., 413`.
   - Wrap `analyze_product_image` in `try...except` to catch unhandled errors and return a sanitized 500 JSON response.
4. **Error Message Sanitization:**
   - In `/api/food/analyze` and `/api/personal-care/analyze`, replace `"errors": [str(exc)]` with sanitized generic client error message: `"errors": ["An unexpected server error occurred. Please try again."]`.
   - Ensure `current_app.logger.error(..., exc_info=True)` is preserved for server-side diagnosis.
5. **Debug Mode Hardening:**
   - Update `app.py` so `debug` is controlled by `FLASK_DEBUG` environment variable, defaulting to `False`.
6. **Health Endpoint:**
   - Implement `GET /health` on the Flask application returning `{"status": "healthy", "app": "PicWise", "version": "1.0.0"}` without invoking OCR.
7. **Production Containerization & Requirements:**
   - Create `Dockerfile` and `.dockerignore`.
   - Update `requirements.txt` with all pinned runtime and WSGI dependencies.

---

## 18. Proposed Test Plan

In Phase 11B Part 2, the following automated tests will be added or executed:

1. **`test_wsgi_entrypoint`:** Verify `from wsgi import application` succeeds and is a valid WSGI callable.
2. **`test_production_config`:** Verify that `create_app()` sets `MAX_CONTENT_LENGTH == 16 * 1024 * 1024`, registers the 413 error handler, and initializes `KNOWLEDGE_BASE`.
3. **`test_health_endpoint`:** Verify `GET /health` returns HTTP 200 with `{"status": "healthy"}` and does not execute OCR.
4. **`test_sanitized_500_error_handling`:** Simulate an unhandled service exception on `/api/food/analyze`, `/api/personal-care/analyze`, and `/api/analyze`; verify HTTP 500 is returned with a generic client error and that `str(exc)` is **not** leaked in the response payload.
5. **`test_legacy_endpoint_oversized_upload`:** Send an upload exceeding 16MB to `POST /api/analyze`; verify it is rejected with HTTP 413.
6. **`test_food_api_e2e_regression`:** Verify `POST /api/food/analyze` processes valid food labels correctly.
7. **`test_personal_care_api_e2e_regression`:** Verify `POST /api/personal-care/analyze` processes valid personal care labels correctly.
8. **`test_safety_semantics_regression`:** Verify unknown ingredients and missing data evaluate to unsafe/unknown and never safe.
9. **Full Regression Suite:** Execute all existing 126 regression tests (Food: 57, Personal Care: 58, Remediation: 11) to confirm zero regressions.

---

## 19. Production Packaging Gap Matrix

| Area | Status | Evidence | Severity | Part 2 Action |
|---|---|---|---|---|
| **Core Application Logic** | PASS | 126/126 tests passing; all safety semantics strictly enforced. | — | None (Frozen). |
| **OCR / OpenMP Lifecycle** | PASS WITH LIMITATIONS | CPU latency 35–110s; sustained batches can accumulate OpenMP thread pools. | P2 | Configure Gunicorn `max_requests=100`, `timeout=180`, `preload_app=False`, and `OMP_NUM_THREADS=2`. |
| **Legacy Upload Guard** | PASS WITH LIMITATIONS | `/api/analyze` lacks explicit pre-read 16MB check present on category routes. | P3 | Add `len(image_bytes) > MAX_IMAGE_SIZE_BYTES` guard returning HTTP 413. |
| **500 Error Sanitization** | PASS WITH LIMITATIONS | `/api/food/analyze` and `/api/personal-care/analyze` return `str(exc)` in JSON payload. | P3 | Sanitize client JSON to generic error message; keep server `logger.error(..., exc_info=True)`. |
| **Debug Mode Guard** | PASS WITH LIMITATIONS | `app.py` runs with `debug=True` when executed directly via `python app.py`. | P3 | Default `debug=False` unless `FLASK_DEBUG=1`; provide dedicated `wsgi.py`. |
| **WSGI Server Configuration** | PASS WITH LIMITATIONS | No `wsgi.py` or `gunicorn.conf.py` exists in the repository. | P3 | Create `wsgi.py` and `gunicorn.conf.py`. |
| **Health / Readiness Route** | PASS WITH LIMITATIONS | No `/health` or `/ready` endpoint exists for container orchestrators. | P3 | Implement lightweight `GET /health` returning HTTP 200 without running OCR. |
| **Containerization Assets** | NOT VERIFIED | No `Dockerfile` or `.dockerignore` exists. | P3 | Create multi-stage/slim `Dockerfile` and `.dockerignore`. |
| **Dependency Panning** | PASS WITH LIMITATIONS | `requirements.txt` lists only 6 packages, omitting `paddleocr`, `pillow`, `opencv`, etc. | P3 | Update `requirements.txt` with all pinned runtime and WSGI dependencies. |
| **Secrets & Security** | PASS | Zero secrets, API keys, or tokens in codebase; `.env` gitignored; purely in-memory image processing. | — | Maintain existing zero-secret architecture. |
| **Filesystem / Write State** | PASS | Zero file writes during request processing; compatible with read-only container root. | — | Maintain in-memory processing; provide writable `/tmp` in container. |
| **Static / Template Serving** | PASS | Flask serves vanilla JS/CSS/templates natively; no node/npm build step required. | — | Retain existing Flask serving. |

---

## 20. Explicit Non-Changes

During Phase 11B Part 1 and the upcoming Part 2, the following components are strictly **frozen** and will **not** be modified:
1. **Food Safety ML Model:** Vectorizer, classifier, hyperparameters, and risk classification logic.
2. **Personal Care ML Models:** Safety, Allergy, and Irritation pipelines and estimators.
3. **Shared PaddleOCR Pipeline:** Architecture, preprocessing variants, line clustering, and confidence scoring.
4. **Food Scoring & Nutrition Methodology:** Nutrient thresholds, scoring algorithms, and formula.
5. **Personal Care Risk Aggregation:** Individual dimension scoring, worst-case aggregation, and display semantics.
6. **Explicit Category Selection:** Strict user-selected category routing (`food` vs `personal_care`).
7. **Failure Semantics:** *Unknown != Safe*, *Missing Data != Safe*, *OCR Failure != Safe*.
8. **Datasets & Knowledge Bases:** Food, nutrition, and personal care CSV/XLSX data files and indexes.
9. **API Response Schemas:** Existing fields and structures for 200 OK responses.

---

## 21. Final Verdict

- **Core Application Readiness:** **PASS**  
  The application logic, machine learning inference, category isolation, OCR extraction, and failure semantics are completely functional, robust, and verified by 126 regression tests.
- **Deployment Readiness:** **PASS WITH LIMITATIONS**  
  The system requires production WSGI configuration with worker recycling (`max_requests=100`), long worker timeout (`timeout=180`), OpenMP thread limits (`OMP_NUM_THREADS=2`), error message sanitization, and a lightweight health endpoint.
- **Packaging Readiness:** **PASS WITH LIMITATIONS**  
  The repository currently lacks a `Dockerfile`, `.dockerignore`, and an exhaustive production `requirements.txt`. These gaps are low-risk and will be resolved in Phase 11B Part 2.
