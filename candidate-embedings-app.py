import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

# Sample candidate profiles
candidates = [
    "Nurse with 5 years of ICU experience, fluent in English and German",
    "Community nurse with home care experience, speaks only Polish",
    "Young nurse after internship, looking for a position in a pediatric hospital",
    "Experienced surgical nurse, worked in operating rooms, fluent in English",
    "Nurse with experience in psychiatry and addiction treatment"
]

# Sample job offers
job_offers = [
    "Regional hospital looking for a surgical nurse to join the operating team",
    "Nursing home hiring a community nurse for in-home patient care",
    "Children’s clinic in Warsaw hiring a junior nurse, training available",
    "Addiction treatment center hiring a nurse with psychiatry experience",
    "University hospital looking for ICU nurse, English required"
]

# Load model (cached to avoid reloading on every refresh)
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Compute embeddings
candidate_embeddings = model.encode(candidates, convert_to_tensor=True)
job_embeddings = model.encode(job_offers, convert_to_tensor=True)

# Page title
st.title("AI Candidate Matching Demo for Nursing Jobs")


# Candidate selection with default 'Please select...'
candidate_options = ["Please select..."] + candidates
selected_candidate = st.selectbox("Select a candidate profile", candidate_options)

# Show results only if a candidate is selected
if selected_candidate != "Please select...":
    index = candidates.index(selected_candidate)
    similarities = util.cos_sim(candidate_embeddings[index], job_embeddings)[0]

    # Find the best matching job offer
    best_match_idx = torch.argmax(similarities).item()
    score = similarities[best_match_idx].item()

    # Sort offers by similarity
    sorted_indices = torch.argsort(similarities, descending=True)

    # Display matching results
    st.subheader("Top Matching Job Offers:")
    for idx in sorted_indices:
        offer = job_offers[idx]
        score = similarities[idx].item()
        st.markdown(f"#{idx+1} offer. **{offer}**")
        st.markdown(f"Match Score: `{score:.2f}`")
        # st.markdown("---")
