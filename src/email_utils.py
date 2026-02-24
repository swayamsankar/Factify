# src/email_utils.py
import re
import validators
import tldextract
import dns.resolver
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import load
import os

# Optional: path to your trained classifier (joblib). If not present we'll skip ML.
CLASSIFIER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'email_detector.joblib')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf_vectorizer.joblib')

# Minimal list of disposable domains (example — extend it)
DISPOSABLE_DOMAINS = {
    "mailinator.com","10minutemail.com","tempmail.com","guerrillamail.com","yopmail.com"
}

SUSPICIOUS_WORDS = [
    "urgent", "verify", "password", "account", "click here", "update", "login", "bank", "secure",
    "immediately", "suspend", "confirm", "pay", "invoice", "winner", "prize"
]

def get_domain(email):
    try:
        return email.split('@',1)[1].lower()
    except Exception:
        return ''

def has_mx_record(domain):
    try:
        answers = dns.resolver.resolve(domain, 'MX')
        return len(answers) > 0
    except Exception:
        return False

def has_spf_record(domain):
    try:
        answers = dns.resolver.resolve(domain, 'TXT')
        for r in answers:
            txt = r.to_text().strip('"')
            if 'v=spf1' in txt.lower():
                return True
    except Exception:
        pass
    return False

def count_links(text):
    urls = re.findall(r'https?://\S+|www\.\S+', text)
    return len(urls), urls

def suspicious_word_score(text):
    text_low = text.lower()
    score = sum(1 for w in SUSPICIOUS_WORDS if w in text_low)
    return score

def is_disposable(domain):
    domain = domain.lower()
    return any(domain.endswith(d) for d in DISPOSABLE_DOMAINS)

def load_optional_ml():
    clf = None
    vec = None
    try:
        if os.path.exists(CLASSIFIER_PATH) and os.path.exists(VECTORIZER_PATH):
            clf = load(CLASSIFIER_PATH)
            vec = load(VECTORIZER_PATH)
    except Exception:
        clf = None
        vec = None
    return clf, vec

# Main checker
def assess_email(from_addr: str, subject: str, body: str):
    """
    Returns: dict with scores and verdict ('fake'|'likely_fake'|'unknown'|'real')
    """
    domain = get_domain(from_addr)
    results = {
        "from": from_addr,
        "domain": domain,
        "mx_ok": False,
        "spf_ok": False,
        "disposable": False,
        "num_links": 0,
        "suspicious_word_count": 0,
        "ml_score": None,
        "verdict": "unknown",
        "reasons": []
    }

    if domain:
        results["mx_ok"] = has_mx_record(domain)
        results["spf_ok"] = has_spf_record(domain)
        if not results["mx_ok"]:
            results["reasons"].append("No MX record for sender domain")
        if not results["spf_ok"]:
            results["reasons"].append("No SPF record for sender domain")

        if is_disposable(domain):
            results["disposable"] = True
            results["reasons"].append("Disposable / temporary email domain")

    # links and suspicious words
    num_links, urls = count_links(subject + " " + body)
    results["num_links"] = num_links
    if num_links > 2:
        results["reasons"].append(f"Multiple links found ({num_links})")

    sw_score = suspicious_word_score(subject + " " + body)
    results["suspicious_word_count"] = sw_score
    if sw_score >= 2:
        results["reasons"].append(f"Suspicious keywords detected ({sw_score})")

    # simple heuristic scoring
    score = 0
    if not results["mx_ok"]:
        score += 2
    if not results["spf_ok"]:
        score += 1
    if results["disposable"]:
        score += 2
    score += min(results["num_links"], 3)
    score += min(sw_score, 3)

    # optional ML model (if present)
    clf, vec = load_optional_ml()
    if clf and vec:
        try:
            X = vec.transform([subject + " " + body])
            prob = clf.predict_proba(X)[0].max()  # probability of predicted class
            pred = clf.predict(X)[0]
            # define ml_score as probability for class 'fake' if classifier is labeled that way
            results["ml_score"] = float(prob)
            if hasattr(clf, "classes_"):
                results["ml_pred"] = str(pred)
            # combine ml output into score
            if str(pred).lower() in ("fake", "phishing", "1"):
                score += 3 * (prob)  # weighted
        except Exception:
            results["ml_score"] = None

    # final verdict thresholds (tweak as you collect data)
    if score >= 6:
        verdict = "fake"
    elif score >= 3:
        verdict = "likely_fake"
    elif score <= 1:
        verdict = "real"
    else:
        verdict = "unknown"

    results["score"] = round(score, 2)
    results["verdict"] = verdict
    results["links"] = urls
    return results
