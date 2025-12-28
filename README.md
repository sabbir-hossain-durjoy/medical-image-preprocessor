# Medical Image Preprocessor — Streamlit App

This repository contains a Streamlit application `app.py` for preprocessing medical images (X-rays, CTs).

Quick steps to run locally:

1. Create a virtual environment and install dependencies:

   python -m venv .venv
   .venv\Scripts\activate; pip install -r requirements.txt

2. Run the app:

   streamlit run app.py

Notes:
- The app supports single image upload and ZIP dataset processing.
- If deploying to Streamlit Cloud, ensure this repo contains `requirements.txt` and `Procfile`.
