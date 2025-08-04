import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

# Load model (cached to avoid reloading on every refresh)
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

tab1, tab2, tab3 = st.tabs(["Demo: Select Candidate", "Demo: Custom Input", "How Embeddings Work"])

with tab1:
    st.title("AI Candidate Matching Demo for Nursing Jobs")

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

    # Candidate selection with default 'Please select...'
    candidate_options = ["Please select..."] + candidates
    selected_candidate = st.selectbox("Select a candidate profile", candidate_options)

    # Show results only if a candidate is selected
    if selected_candidate != "Please select...":
        candidate_embeddings = model.encode(candidates, convert_to_tensor=True)
        job_embeddings = model.encode(job_offers, convert_to_tensor=True)
        index = candidates.index(selected_candidate)
        similarities = util.cos_sim(candidate_embeddings[index], job_embeddings)[0]

        # Sort offers by similarity
        sorted_indices = torch.argsort(similarities, descending=True)

        # Display matching results
        st.subheader("Top Matching Job Offers:")
        for idx in sorted_indices:
            offer = job_offers[idx]
            score = similarities[idx].item()
            st.markdown(f"#{idx+1} offer. **{offer}**")
            st.markdown(f"Match Score: `{score:.2f}`")

with tab2:
    st.title("AI Candidate-to-Job Matching Demo (Custom Input)")
    st.markdown("This tool uses semantic embeddings to recommend the best job matches for nursing candidates based on profile descriptions.")

    # Text areas for user input
    st.subheader("Step 1: Enter your candidate profiles")
    candidates_input = st.text_area(
        "Enter one candidate profile per line:",
        "Nurse with 5 years of ICU experience, fluent in English and German\n"
        "Community nurse with home care experience, speaks only Polish\n"
        "Young nurse after internship, looking for a position in a pediatric hospital"
    )

    st.subheader("Step 2: Enter your job offers")
    jobs_input = st.text_area(
        "Enter one job offer per line:",
        "Regional hospital looking for a surgical nurse to join the operating team\n"
        "Nursing home hiring a community nurse for in-home patient care\n"
        "Children’s clinic in Warsaw hiring a junior nurse, training available"
    )

    # Parse text into lists
    candidates = [line.strip() for line in candidates_input.strip().split("\n") if line.strip()]
    job_offers = [line.strip() for line in jobs_input.strip().split("\n") if line.strip()]

    # Matching logic
    if st.button("Run Matching"):
        if not candidates or not job_offers:
            st.error("Please enter at least one candidate and one job offer.")
        else:
            candidate_embeddings = model.encode(candidates, convert_to_tensor=True)
            job_embeddings = model.encode(job_offers, convert_to_tensor=True)

            st.subheader("Matching Results")

            for i, candidate in enumerate(candidates):
                similarities = util.cos_sim(candidate_embeddings[i], job_embeddings)[0]
                sorted_indices = torch.argsort(similarities, descending=True)

                st.markdown(f"### Candidate {i+1}:")
                st.markdown(f"`{candidate}`")

                for rank, idx in enumerate(sorted_indices[:3]):
                    offer = job_offers[idx]
                    score = similarities[idx].item()
                    st.markdown(f"**Match #{rank + 1}:** {offer}")
                    st.markdown(f"_Score: `{score:.2f}`_")

with tab3:
    st.title("How Semantic Embeddings Work")
    st.markdown("""
Semantic embeddings are vector representations of text that capture the meaning and context of sentences. Using models like Sentence Transformers, each candidate profile and job offer is converted into a high-dimensional vector. The similarity between these vectors (typically measured by cosine similarity) reflects how closely the meanings of the texts match, regardless of exact wording.

For example, "Nurse with ICU experience" and "ICU nurse wanted" will have very similar embeddings, even though the wording is different.

### Business Usefulness

Semantic matching with embeddings can be used in many business scenarios, such as:
- **Recruitment & HR:** Automatically match candidate profiles to job descriptions, reducing manual screening time and improving the quality of recommendations.
- **Customer Support:** Route support tickets to the most relevant department or agent based on the semantic content of the request.
- **Document Search:** Find the most relevant documents, FAQs, or knowledge base articles for a user's query, even if the query uses different words than the documents.
- **Personalization:** Recommend products, services, or content based on the semantic similarity between user preferences and available options.

This approach enables smarter, context-aware automation and recommendations, leading to better user experience and business efficiency.
""")
