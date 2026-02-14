# Movie Prediction — Flask App

Simple web app that demonstrates a movie-prediction demo using prepackaged models and the TMDB datasets.

**Quick Links**
- App entry: [app.py](app.py)
- Dependencies: [requirements.txt](requirements.txt)
- Dockerfile: [Dockerfile](Dockerfile)
- Data folder: [data/](data/)

## Requirements
- Python 3.11+ (recommended)
- The Python dependencies are listed in [requirements.txt](requirements.txt).

## Quick start (Docker)
Build the image from the project root and run a container exposing port 5000:

```bash
docker build -t movie-prediction:latest .
docker run --rm -p 5000:5000 movie-prediction:latest
```

Open http://localhost:5000 in your browser.

## Run locally (no Docker)
Create a virtual environment, install dependencies, then run the app:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
python app.py
```

The app listens on port 5000 by default.

## Data
This repo includes the TMDB CSVs under the [data/](data/) directory. The app expects these files to be present if you want to inspect or rebuild models.

## Production notes
- The provided `Dockerfile` is intentionally minimal for quick builds. For production consider using a production WSGI server (e.g., `gunicorn`) and applying multi-stage builds to reduce image size.
- If you want, I can update the Dockerfile to run `gunicorn --workers 3 "app:app"` and add a healthcheck.

## License
See the project `LICENSE` file.