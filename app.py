import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up page configuration
st.set_page_config(page_title="AI Resume Matcher", page_icon="💼", layout="centered")

st.title("💼 AI-Powered Resume & Job Description Matcher")
st.write("Optimize your resume for applicant tracking systems (ATS) using Machine Learning.")

# 1. UI Elements for Inputs
st.subheader("1. Paste the Target Job Description")
job_description = st.text_area("Enter the job requirements here:", height=200, placeholder="Looking for a Software Engineer with skills in Python, Git, and Database Management...")

st.subheader("2. Upload Your Resume")
uploaded_file = st.file_uploader("Choose your resume file (PDF format only)", type=["pdf"])

# 2. Helper function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    extracted_text = ""
    for page in pdf_reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + " "
    return extracted_text.strip()

# 3. Main Processing Logic
if st.button("Analyze Match Percentage", type="primary"):
    if job_description and uploaded_file:
        with st.spinner("Processing text and running NLP models..."):
            # Extract resume text
            resume_text = extract_text_from_pdf(uploaded_file)
            
            if resume_text:
                # Combine documents for vectorization
                documents = [resume_text, job_description]
                
                # Initialize TF-IDF Vectorizer
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(documents)
                
                # Calculate Cosine Similarity
                similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
                match_percentage = round(similarity_matrix[0][0] * 100, 2)
                
                # Display Results
                st.success("Analysis Complete!")
                st.metric(label="ATS Match Score", value=f"{match_percentage}%")
                
                if match_percentage >= 75:
                    st.balloons()
                    st.write("🟢 **Excellent Match!** Your profile aligns well with the key requirements.")
                elif match_percentage >= 50:
                    st.write("🟡 **Good Start.** Consider adding more matching keywords from the job description to boost your score.")
                else:
                    st.write("🔴 **Low Match Score.** You might be missing core skills or formatting details required for this role.")
            else:
                st.error("Could not read text from the uploaded PDF. Ensure the file is not scanned/an image.")
    else:
        st.warning("Please provide both a job description and a resume file to run the analysis.")
