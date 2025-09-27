# AI-Powered Resume Ranker

## 📌 Overview
This project implements an **AI-Powered Resume Ranker** that automatically evaluates and ranks resumes against a given **Job Description (JD)**.  
It simulates a simplified **Applicant Tracking System (ATS)** by using **TF-IDF vectorization**, **cosine similarity**, **skill extraction**, and **experience heuristics**.  

The system outputs a ranked CSV file with candidate details, matched skills, similarity scores, and a final ranking.

---

## ⚙️ Tools & Libraries
- **Python 3.10+**
- **scikit-learn** → TF-IDF, cosine similarity
- **pandas, numpy** → data handling
- **pdfminer.six** → PDF text extraction
- **regex** → skill & experience extraction
- **joblib** → model saving
- *(optional)* Streamlit → simple UI demo

---

## 📂 Project Structure
ai-resume-ranker/
│── utils.py # Helper functions (text cleaning, PDF parsing, regex)
│── ranker.py # Core ranking logic (TF-IDF, similarity, scoring)
│── main.py # CLI entry point
│── requirements.txt # Dependencies
│── jd.txt # Sample Job Description
│── resumes_sample/ # Sample resumes (.txt or .pdf)
│── artifacts/ # Output folder (ranking.csv, vectorizer, jd_skills.json)
│── README.md # Project documentation

Sample Output
Candidate	Similarity	SkillsMatched	SkillsRequired	MatchedSkillsList	ExpYears	SkillMatch%	FinalScore
resume_arya.txt	0.78	5	6	python, tensorflow, pytorch, docker, aws	2	83.3%	0.8421
resume_sam.txt	0.65	4	6	python, scikit-learn, docker, aws	3	66.7%	0.7103
resume_alex.txt	0.28	1	6	python	4	16.7%	0.3210
🚀 Future Improvements

Use Sentence-BERT embeddings for semantic matching.

Improve skill extraction using Named Entity Recognition (NER).

Add a Streamlit web app for interactive use.

Support scanned PDF resumes using OCR.

✨ Author

Arya Katulwar
Developed as part of the Elevate Labs AI/ML Internship Project
