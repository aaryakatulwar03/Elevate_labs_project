# utils.py
import re
import io
from pdfminer.high_level import extract_text
import regex as re2

def clean_text(text: str) -> str:
    """Basic text cleanup"""
    if not isinstance(text, str):
        return ""
    t = text.replace("\x00", " ")
    t = t.encode("utf-8", errors="ignore").decode("utf-8")
    t = re.sub(r"\s+", " ", t)
    t = t.strip()
    return t

def extract_text_from_pdf(path_or_bytes) -> str:
    """Extract text from PDF file"""
    try:
        if isinstance(path_or_bytes, (bytes, bytearray)):
            return clean_text(extract_text(io.BytesIO(path_or_bytes)))
        else:
            return clean_text(extract_text(path_or_bytes))
    except Exception as e:
        print(f"[utils.extract_text_from_pdf] extraction failed: {e}")
        return ""

def detect_skills(text: str, skills_vocab: set):
    """Find skills from a given skills list"""
    t = text.lower()
    found = set()
    for sk in skills_vocab:
        sk = sk.lower().strip()
        if not sk:
            continue
        pat = r"(?:^|\W)" + re2.escape(sk) + r"(?:$|\W)"
        if re2.search(pat, t, flags=re2.IGNORECASE):
            found.add(sk)
    return sorted(found)

def extract_experience_years(text: str) -> float:
    """Extract number of years of experience from resume text"""
    t = text.lower()
    years = []

    # e.g., "3 years", "5+ yrs"
    for m in re.finditer(r"(\d{1,2})\s*\+?\s*(?:years|yrs|year|yr)\b", t):
        try:
            years.append(int(m.group(1)))
        except:
            pass

    # e.g., "2018-2021"
    for m in re.finditer(r"(19|20)\d{2}\s*[-–]\s*(19|20)\d{2}", t):
        try:
            parts = re.split(r"[-–]", m.group(0))
            a, b = int(parts[0]), int(parts[1])
            if b >= a:
                years.append(b - a)
        except:
            pass

    if len(years) == 0:
        return 0.0
    years = sorted(years)
    mid = years[len(years)//2]
    return float(min(30.0, mid))
