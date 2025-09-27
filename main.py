# main.py
import argparse
import glob
import os
import json
import joblib
from ranker import rank_resumes

def load_skills(path):
    """Load skills list from file or default set"""
    if not path:
        return set([
            "python","java","c++","sql","javascript","tensorflow","pytorch","scikit-learn",
            "machine learning","deep learning","data science","aws","gcp","docker","kubernetes",
            "nlp","opencv","pandas","numpy","streamlit","flask"
        ])
    with open(path, 'r', encoding='utf-8') as f:
        return set([line.strip().lower() for line in f if line.strip()])

def main():
    parser = argparse.ArgumentParser(description="AI Resume Ranker CLI")
    parser.add_argument("--jd", type=str, required=True, help="Path to JD text file")
    parser.add_argument("--resumes", type=str, required=True, help="Folder with resumes")
    parser.add_argument("--skills", type=str, default=None, help="Optional skills list (.txt)")
    parser.add_argument("--out", type=str, default="artifacts/ranking.csv", help="Output CSV path")
    parser.add_argument("--weights", type=float, nargs=3, default=(0.6,0.3,0.1),
                        help="Weights for sim, skill, exp")
    args = parser.parse_args()

    # Load job description
    with open(args.jd, 'r', encoding='utf-8') as f:
        jd_text = f.read()

    # Load resumes
    resumes = glob.glob(os.path.join(args.resumes, "*"))
    skills = load_skills(args.skills)

    # Ensure output folder exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Run ranking
    df, vectorizer, jd_skills = rank_resumes(jd_text, resumes, skills, weights=tuple(args.weights))
    df.to_csv(args.out, index=False)
    joblib.dump(vectorizer, "artifacts/tfidf_vectorizer.joblib")
    with open("artifacts/jd_skills.json", "w", encoding="utf-8") as f:
        json.dump(sorted(list(jd_skills)), f, indent=2)

    print(f"[main] Ranking saved to {args.out}")

if __name__ == "__main__":
    main()
