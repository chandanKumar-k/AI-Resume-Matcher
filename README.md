# AI-Powered Resume & Job Description Matcher 💼

A production-ready Applicant Tracking System (ATS) simulation dashboard built with **Python** and **Streamlit** to mathematically analyze and score textual alignment using Machine Learning pipelines.

## 🚀 Architectural Core & NLP Mechanics
* **Text Extraction Engine**: Leverages `pypdf` to programmatically parse, sanitize, and extract raw unstructured string arrays out of multi-page PDF documents.
* **Vectorization Protocol (TF-IDF)**: Utilizes `scikit-learn`'s `TfidfVectorizer` to filter out common grammatical English stop words and map domain-specific technical terms into multidimensional numerical weight matrices based on keyword importance.
* **Proximity Calculation (Cosine Similarity)**: Computes the geometric cosine angle between the computed Resume Vector and the Job Description Vector to generate an objective, accurate ATS match percentage between `0%` and `100%`.

## 🛠️ Tech Stack
* **Language**: Python 3
* **AI/ML Libraries**: Scikit-Learn (NLP Vectorization)
* **File Processing**: PyPDF
* **User Interface**: Streamlit Dashboard Engine

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   ```

2. **Navigate into the project directory:**
   ```bash
   cd AI-Resume-Matcher
   ```

3. **Install the required dependencies:**
   ```bash
   python -m pip install streamlit scikit-learn pypdf
   ```

4. **Launch the Streamlit web server:**
   ```bash
   python -m streamlit run app.py
   ```
