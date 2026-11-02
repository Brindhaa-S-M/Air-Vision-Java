# AirVision Java Backend (hybrid Java + Python)

This Spring Boot application replaces the old `backend/main.py` API server. The existing `backend/ml.py`, trained Random Forest model, datasets, and Python data-processing scripts remain unchanged.

Java provides the REST API and serves the existing `frontend/` files. For ML operations, Java starts the installed Python interpreter and calls the existing functions in `backend.ml`.

## Requirements
- JDK 17 or newer
- Maven 3.6+
- Python 3.10 or newer
- Python packages from the project's root `requirements.txt`

## Run on Windows PowerShell
Open a terminal in the `backend-java` folder:

```powershell
mvn spring-boot:run
```

If the Python command on your computer is not `python`, change `airvision.python-command` in `src/main/resources/application.properties` (for example, to `py -3` is not accepted as one command; use `python` on PATH or configure a wrapper executable).

Before starting Java, install the existing Python requirements from the project root:

```powershell
cd ..
python -m pip install -r requirements.txt
cd backend-java
mvn spring-boot:run
```

Open `http://localhost:8000`. The API endpoints are unchanged:
- `GET /api/health`
- `GET /api/stations`
- `GET /api/metrics`
- `POST /api/train`
- `POST /api/predict`
- `GET /api/prediction-grid`

The original `backend/main.py` is intentionally retained as a backup in this project package. Do not run Uvicorn and Spring Boot on port 8000 at the same time.
