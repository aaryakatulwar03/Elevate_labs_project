# ranker.py
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import clean_text, extract_text_from_pdf, detect_skills, extract_experience_years

def normalize_series(s: pd.Series) -> pd.Series:
    """Normalize values to 0–1 scale"""
    if s.empty:
        return s
    arr = s.astype(float).to_numpy()
    lo, hi = arr.min(), arr.max()
    if abs(hi - lo) < 1e-9:
        return pd.Series(np.ones_like(arr))
    return pd.Series((arr - lo) / (hi - lo))

def rank_resumes(jd_text: str, resume_paths: list, skills_vocab: set,
                 weights=(0.6, 0.3, 0.1), min_df=1, max_features=100000):
    """Main ranking function"""
    docs = [jd_text]
    names = ["__JD__"]
    raw_resume_texts = []

    # Load resumes
    for p in resume_paths:
        text = ""
        ext = os.path.splitext(p)[1].lower()
        if ext == '.pdf':
            text = extract_text_from_pdf(p)
        elif ext == '.txt':
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        else:
            text = extract_text_from_pdf(p)

        raw = clean_text(text)
        docs.append(raw)
        names.append(os.path.basename(p))
        raw_resume_texts.append(raw)

    # TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2),
                                 min_df=min_df, max_features=max_features)
    X = vectorizer.fit_transform(docs)
    jd_vec = X[0:1]
    res_vecs = X[1:]
    sims = cosine_similarity(res_vecs, jd_vec).ravel()

    # Skills
    jd_skills = set(detect_skills(jd_text, skills_vocab))
    if len(jd_skills) == 0:
        jd_skills = set(list(skills_vocab)[:20])

    skill_counts = []
    matched_skill_lists = []
    for txt in raw_resume_texts:
        matched = detect_skills(txt, jd_skills)
        skill_counts.append(len(matched))
        matched_skill_lists.append(matched)

    # Experience
    exp_years = [extract_experience_years(txt) for txt in raw_resume_texts]

    # Build DataFrame
    df = pd.DataFrame({
        "Candidate": names[1:],
        "Similarity": sims,
        "SkillsMatched": skill_counts,
        "SkillsRequired": [len(jd_skills)] * len(skill_counts),
        "MatchedSkillsList": [", ".join(lst) for lst in matched_skill_lists],
        "ExpYears": exp_years
    })

    # Normalization
    sim_norm = normalize_series(df["Similarity"])
    skill_pct = df["SkillsMatched"] / df["SkillsRequired"].replace(0, np.nan)
    skill_pct = skill_pct.fillna(0.0)
    skill_norm = normalize_series(skill_pct)
    exp_norm = normalize_series(df["ExpYears"])

    # Weighted scoring
    w_sim, w_skill, w_exp = weights
    final = w_sim * sim_norm + w_skill * skill_norm + w_exp * exp_norm

    # Final dataframe
    df["SkillMatch%"] = (skill_pct * 100).round(1)
    df["FinalScore"] = final.round(4)
    df_sorted = df.sort_values("FinalScore", ascending=False).reset_index(drop=True)
    df_sorted["Similarity"] = df_sorted["Similarity"].round(4)
    df_sorted["ExpYears"] = df_sorted["ExpYears"].round(1)
    return df_sorted, vectorizer, jd_skills
