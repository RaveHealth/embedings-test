# AI Candidate Matching Demo


This is a Streamlit application for matching candidates to job offers using semantic embeddings (Sentence Transformers).

## How Semantic Embeddings Work

Semantic embeddings are vector representations of text that capture the meaning and context of sentences. Using models like Sentence Transformers, each candidate profile and job offer is converted into a high-dimensional vector. The similarity between these vectors (typically measured by cosine similarity) reflects how closely the meanings of the texts match, regardless of exact wording.

For example, "Nurse with ICU experience" and "ICU nurse wanted" will have very similar embeddings, even though the wording is different.

## Business Usefulness

Semantic matching with embeddings can be used in many business scenarios, such as:
- **Recruitment & HR:** Automatically match candidate profiles to job descriptions, reducing manual screening time and improving the quality of recommendations.
- **Customer Support:** Route support tickets to the most relevant department or agent based on the semantic content of the request.
- **Document Search:** Find the most relevant documents, FAQs, or knowledge base articles for a user's query, even if the query uses different words than the documents.
- **Personalization:** Recommend products, services, or content based on the semantic similarity between user preferences and available options.

This approach enables smarter, context-aware automation and recommendations, leading to better user experience and business efficiency.

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

Deployed to https://rave-embedings.streamlit.app/

## Files
- `app.py` – main Streamlit app with two tabs
- `requirements.txt` – required packages
- `candidate-embedings.ipynb`, `sentence-transfomers.ipynb` – example notebooks

## Author
RaveHealth / Wiesław