# AI Candidate Matching Demo

This is a Streamlit application for matching candidates to job offers using semantic embeddings (Sentence Transformers).

## Features
- Compare candidate profiles with job offers
- Two modes: ready-made examples and custom user data
- Results sorted by semantic similarity

## Local Run

1. Create and activate a virtual environment (optional):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deployment 

We need decide where to deploy

## Files
- `app.py` – main Streamlit app with two tabs
- `requirements.txt` – required packages
- `candidate-embedings.ipynb`, `sentence-transfomers.ipynb` – example notebooks

## Author
RaveHealth / Wiesław