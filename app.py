from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import re
import math
from urllib.parse import urlparse
 
app = Flask(__name__)
 
# ================================================================
#  LOAD URL MODEL FILES (5 files) — unchanged
# ================================================================
model           = joblib.load("phishguard_model.pkl")
le              = joblib.load("phishguard_label_encoder.pkl")
CLASS_METRICS   = joblib.load("phishguard_metrics.pkl")
explainer       = joblib.load("phishguard_shap_explainer.pkl")
FEATURE_COLUMNS = joblib.load("phishguard_feature_columns.pkl")
 
# ================================================================
#  LOAD EMAIL MODEL FILES (4 files) — NEW
# ================================================================
email_model      = joblib.load("email_model.pkl")
email_vectorizer = joblib.load("email_vectorizer.pkl")
email_le         = joblib.load("email_label_encoder.pkl")
EMAIL_METRICS    = joblib.load("email_metrics.pkl")
 
print("✅ All models loaded successfully")
 
# ================================================================
#  EXACT FEATURE COLUMNS FROM YOUR NEW PKL:
# ['url_length', 'hostname_length', 'path_length', 'query_length',
#  'fragment_length', 'count_digits', 'count_letters', 'count_dots',
#  'count_hyphens', 'count_underscores', 'count_slashes', 'count_question',
#  'count_equal', 'count_at', 'count_ampersand', 'count_percent',
#  'count_hash', 'digit_letter_ratio', 'digit_length_ratio',
#  'num_subdomains', 'has_ip', 'has_port', 'hostname_digits',
#  'hostname_hyphens', 'hostname_length2', 'suspicious_tld',
#  'has_risky_ext', 'is_shortened', 'url_depth', 'num_params',
#  'has_redirect', 'has_at_symbol', 'url_entropy', 'hostname_entropy',
#  'num_tokens', 'longest_token', 'shortest_token', 'keyword_count',
#  'has_login', 'has_verify', 'has_secure', 'has_account', 'has_update',
#  'has_paypal', 'has_bank', 'has_free', 'has_signin', 'has_confirm',
#  'has_password', 'has_suspend']
# ================================================================
 
SUSPICIOUS_TLDS  = ['tk','ml','ga','cf','gq','xyz','top','pw','cc',
                    'ru','vn','info','online','site','website','club']
RISKY_EXTENSIONS = ['.exe','.zip','.rar','.bat','.sh','.apk',
                    '.jar','.msi','.ps1','.dmg','.iso']
URL_SHORTENERS   = ['bit.ly','tinyurl','goo.gl','t.co','ow.ly',
                    'is.gd','buff.ly','short.link']
PHISH_KEYWORDS   = [
    'login','verify','secure','update','account','bank','free',
    'bonus','signin','confirm','password','credential','wallet',
    'paypal','ebay','amazon','billing','support','alert','suspend',
    'unlock','validate','authorize','recover','unusual'
]
 
# ================================================================
#  HELPER FUNCTIONS — unchanged
# ================================================================
 
def safe_parse(url):
    try:
        return urlparse(url if '://' in url else 'http://' + url)
    except:
        return urlparse('http://invalid.com')
 
def url_entropy(url):
    if not url: return 0.0
    prob = [url.count(c)/len(url) for c in set(url)]
    return round(-sum(p * math.log2(p) for p in prob), 4)
 
def has_ip(url):
    return 1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', url) else 0
 
def safe_port(url):
    try:
        return 1 if safe_parse(url).port else 0
    except:
        return 0
 
def get_tld(url):
    try:
        parts = safe_parse(url).netloc.split('.')
        return parts[-1].lower() if parts else ''
    except:
        return ''
 
# ================================================================
#  FEATURE EXTRACTION — unchanged
# ================================================================
 
def build_features(input_df):
    f    = pd.DataFrame()
    urls = input_df['url'].apply(str)
 
    f['url_length']           = urls.apply(len)
    f['hostname_length']      = urls.apply(lambda x: len(safe_parse(x).netloc))
    f['path_length']          = urls.apply(lambda x: len(safe_parse(x).path))
    f['query_length']         = urls.apply(lambda x: len(safe_parse(x).query))
    f['fragment_length']      = urls.apply(lambda x: len(safe_parse(x).fragment))
    f['count_digits']         = urls.apply(lambda x: sum(c.isdigit() for c in x))
    f['count_letters']        = urls.apply(lambda x: sum(c.isalpha() for c in x))
    f['count_dots']           = urls.apply(lambda x: x.count('.'))
    f['count_hyphens']        = urls.apply(lambda x: x.count('-'))
    f['count_underscores']    = urls.apply(lambda x: x.count('_'))
    f['count_slashes']        = urls.apply(lambda x: x.count('/'))
    f['count_question']       = urls.apply(lambda x: x.count('?'))
    f['count_equal']          = urls.apply(lambda x: x.count('='))
    f['count_at']             = urls.apply(lambda x: x.count('@'))
    f['count_ampersand']      = urls.apply(lambda x: x.count('&'))
    f['count_percent']        = urls.apply(lambda x: x.count('%'))
    f['count_hash']           = urls.apply(lambda x: x.count('#'))
    f['digit_letter_ratio']   = f['count_digits'] / (f['count_letters'] + 1)
    f['digit_length_ratio']   = f['count_digits'] / (f['url_length'] + 1)
    f['num_subdomains']       = urls.apply(lambda x: max(0, len(safe_parse(x).netloc.split('.')) - 2))
    f['has_ip']               = urls.apply(has_ip)
    f['has_port']             = urls.apply(safe_port)
    f['hostname_digits']      = urls.apply(lambda x: sum(c.isdigit() for c in safe_parse(x).netloc))
    f['hostname_hyphens']     = urls.apply(lambda x: safe_parse(x).netloc.count('-'))
    f['hostname_length2']     = urls.apply(lambda x: len(safe_parse(x).netloc))
    f['suspicious_tld']       = urls.apply(lambda x: 1 if get_tld(x) in SUSPICIOUS_TLDS else 0)
    f['has_risky_ext']        = urls.apply(lambda x: 1 if any(x.lower().endswith(e) for e in RISKY_EXTENSIONS) else 0)
    f['is_shortened']         = urls.apply(lambda x: 1 if any(s in x for s in URL_SHORTENERS) else 0)
    f['url_depth']            = urls.apply(lambda x: x.count('/'))
    f['num_params']           = urls.apply(lambda x: len(safe_parse(x).query.split('&')) if safe_parse(x).query else 0)
    f['has_redirect']         = urls.apply(lambda x: 1 if '//' in x else 0)
    f['has_at_symbol']        = urls.apply(lambda x: 1 if '@' in x else 0)
    f['url_entropy']          = urls.apply(url_entropy)
    f['hostname_entropy']     = urls.apply(lambda x: url_entropy(safe_parse(x).netloc))
 
    def tokens(url):
        t = [x for x in re.split(r'[./?=&\-_]', url) if x]
        if not t: return pd.Series([0, 0, 0])
        return pd.Series([len(t), max(len(x) for x in t), min(len(x) for x in t)])
 
    f[['num_tokens','longest_token','shortest_token']] = urls.apply(tokens)
 
    f['keyword_count']        = urls.apply(lambda x: sum(w in x.lower() for w in PHISH_KEYWORDS))
    f['has_login']            = urls.apply(lambda x: 1 if 'login'    in x.lower() else 0)
    f['has_verify']           = urls.apply(lambda x: 1 if 'verify'   in x.lower() else 0)
    f['has_secure']           = urls.apply(lambda x: 1 if 'secure'   in x.lower() else 0)
    f['has_account']          = urls.apply(lambda x: 1 if 'account'  in x.lower() else 0)
    f['has_update']           = urls.apply(lambda x: 1 if 'update'   in x.lower() else 0)
    f['has_paypal']           = urls.apply(lambda x: 1 if 'paypal'   in x.lower() else 0)
    f['has_bank']             = urls.apply(lambda x: 1 if 'bank'     in x.lower() else 0)
    f['has_free']             = urls.apply(lambda x: 1 if 'free'     in x.lower() else 0)
    f['has_signin']           = urls.apply(lambda x: 1 if 'signin'   in x.lower() else 0)
    f['has_confirm']          = urls.apply(lambda x: 1 if 'confirm'  in x.lower() else 0)
    f['has_password']         = urls.apply(lambda x: 1 if 'password' in x.lower() else 0)
    f['has_suspend']          = urls.apply(lambda x: 1 if 'suspend'  in x.lower() else 0)
 
    return f
 
# ================================================================
#  CLEAN URL — unchanged
# ================================================================
 
def clean_url(url):
    url = str(url).strip().lower()
    url = re.sub(r'^https?://', '', url)
    url = re.sub(r'^www\.', '', url)
    return url
 
# ================================================================
#  FEATURE LABELS — unchanged
# ================================================================
 
FEATURE_LABELS = {
    'url_length'        : 'URL Length',
    'hostname_length'   : 'Hostname Length',
    'hostname_length2'  : 'Hostname Length',
    'path_length'       : 'Path Length',
    'url_entropy'       : 'URL Randomness (Entropy)',
    'hostname_entropy'  : 'Hostname Randomness',
    'num_subdomains'    : 'Number of Subdomains',
    'has_ip'            : 'Contains Raw IP Address',
    'suspicious_tld'    : 'Suspicious Domain Extension',
    'has_risky_ext'     : 'Risky File Extension (.exe/.apk...)',
    'is_shortened'      : 'URL Shortener Detected',
    'has_redirect'      : 'Redirect Trick Detected',
    'has_at_symbol'     : '@ Symbol in URL',
    'keyword_count'     : 'Suspicious Keyword Count',
    'has_login'         : 'Contains "login"',
    'has_verify'        : 'Contains "verify"',
    'has_paypal'        : 'Contains "paypal"',
    'has_bank'          : 'Contains "bank"',
    'has_secure'        : 'Contains "secure"',
    'has_account'       : 'Contains "account"',
    'has_password'      : 'Contains "password"',
    'has_suspend'       : 'Contains "suspend"',
    'has_signin'        : 'Contains "signin"',
    'has_confirm'       : 'Contains "confirm"',
    'has_free'          : 'Contains "free"',
    'has_update'        : 'Contains "update"',
    'digit_letter_ratio': 'Digit-to-Letter Ratio',
    'digit_length_ratio': 'Digit-to-Length Ratio',
    'hostname_digits'   : 'Digits in Hostname',
    'hostname_hyphens'  : 'Hyphens in Hostname',
    'has_port'          : 'Non-Standard Port Used',
    'num_params'        : 'Query Parameter Count',
    'count_dots'        : 'Dot Count',
    'count_hyphens'     : 'Hyphen Count',
    'url_depth'         : 'URL Depth',
    'longest_token'     : 'Longest Word in URL',
    'count_at'          : '@ Symbol Count',
    'count_percent'     : 'Percent Encoding Count',
    'count_hash'        : 'Hash Symbol Count',
}
 
# ================================================================
#  URL PREDICT FUNCTION — unchanged
# ================================================================
 
def predict_url(raw_url):
    cleaned = clean_url(raw_url)
 
    # fix for bare domains
    if '/' not in cleaned:
        cleaned = cleaned + '/'
 
    feat_df = build_features(pd.DataFrame({'url': [cleaned]}))
    feat_df = feat_df[FEATURE_COLUMNS]
 
    pred_idx       = int(model.predict(feat_df)[0])
    label          = le.inverse_transform([pred_idx])[0]
    proba_all      = model.predict_proba(feat_df)[0]
    confidence_pct = round(float(proba_all[pred_idx]) * 100, 1)
 
    all_proba = {
        cls: round(float(proba_all[i]) * 100, 1)
        for i, cls in enumerate(le.classes_)
    }
 
    metrics = CLASS_METRICS[label]
 
    shap_vals      = explainer(feat_df)
    shap_for_class = shap_vals.values[0, :, pred_idx]
 
    shap_pairs = sorted(
        zip(FEATURE_COLUMNS, shap_for_class),
        key=lambda x: abs(x[1]),
        reverse=True
    )
 
    feat_values = feat_df.iloc[0].to_dict()
    top_reasons = []
 
    for feat_name, shap_val in shap_pairs[:6]:
        actual_val = feat_values[feat_name]
        direction  = "increased" if shap_val > 0 else "decreased"
        label_name = FEATURE_LABELS.get(feat_name, feat_name.replace('_', ' ').title())
 
        if actual_val in [0, 1] and any(feat_name.startswith(p) for p in ('has_', 'is_', 'suspicious_')):
            display_val = "Yes" if actual_val == 1 else "No"
        else:
            display_val = str(round(actual_val, 3) if isinstance(actual_val, float) else int(actual_val))
 
        top_reasons.append({
            "feature"   : label_name,
            "value"     : display_val,
            "shap"      : round(float(shap_val), 3),
            "direction" : direction,
            "impact"    : "high"   if abs(shap_val) > 0.3 else
                          "medium" if abs(shap_val) > 0.1 else "low",
        })
 
    return {
        "url"            : raw_url,
        "label"          : label,
        "confidence_pct" : confidence_pct,
        "all_proba"      : all_proba,
        "metrics"        : metrics,
        "top_reasons"    : top_reasons,
        "mode"           : "url",
    }
 
# ================================================================
#  EMAIL TEXT CLEANING — NEW
# ================================================================
 
def clean_email_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' urltoken ', text)
    text = re.sub(r'\S+@\S+', ' emailtoken ', text)
    text = re.sub(r'\d+', ' numtoken ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
 
# ================================================================
#  EMAIL PREDICT FUNCTION — NEW
# ================================================================
 
def predict_email(raw_email_text):
    cleaned    = clean_email_text(raw_email_text)
    vectorized = email_vectorizer.transform([cleaned])
 
    pred_idx       = int(email_model.predict(vectorized)[0])
    label          = email_le.inverse_transform([pred_idx])[0]
    proba_all      = email_model.predict_proba(vectorized)[0]
    confidence_pct = round(float(proba_all[pred_idx]) * 100, 1)
 
    all_proba = {
        cls: round(float(proba_all[i]) * 100, 1)
        for i, cls in enumerate(email_le.classes_)
    }
 
    metrics = EMAIL_METRICS[label]
 
    # top keywords that influenced the decision
    feature_names = email_vectorizer.get_feature_names_out()
    tfidf_scores  = vectorized.toarray()[0]
    top_indices   = tfidf_scores.argsort()[::-1][:6]
 
    top_reasons = []
    for idx in top_indices:
        if tfidf_scores[idx] > 0:
            top_reasons.append({
                "feature"   : f'Word: "{feature_names[idx]}"',
                "value"     : round(float(tfidf_scores[idx]), 3),
                "direction" : "increased" if label == "Phishing Email" else "decreased",
                "impact"    : "high"   if tfidf_scores[idx] > 0.3 else
                              "medium" if tfidf_scores[idx] > 0.1 else "low",
            })
 
    return {
        "email_text"     : raw_email_text[:100] + "..." if len(raw_email_text) > 100 else raw_email_text,
        "label"          : label,
        "confidence_pct" : confidence_pct,
        "all_proba"      : all_proba,
        "metrics"        : metrics,
        "top_reasons"    : top_reasons,
        "mode"           : "email",
    }
 
# ================================================================
#  FLASK ROUTES
# ================================================================
 
@app.route("/", methods=["GET", "POST"])
def home():
    report = None
    error  = None
 
    if request.method == "POST":
        mode = request.form.get("mode", "url")
 
        if mode == "email":
            email_text = request.form.get("email_text", "").strip()
            if email_text:
                try:
                    report = predict_email(email_text)
                except Exception as e:
                    error = f"Could not analyse this email: {str(e)}"
            else:
                error = "Please paste some email text to analyse."
        else:
            url = request.form.get("url", "").strip()
            if url:
                try:
                    report = predict_url(url)
                except Exception as e:
                    error = f"Could not analyse this URL: {str(e)}"
 
    return render_template("index.html", report=report, error=error)
 
 
@app.route("/how")
def how():
    return render_template("how.html")
 
 
@app.route("/about")
def about():
    return render_template("about.html")
 
 
# ================================================================
#  REST API ENDPOINT — updated to support both url and email
# ================================================================
 
@app.route("/api/scan", methods=["POST"])
def api_scan():
    data = request.get_json()
 
    if not data:
        return jsonify({"error": "Please send JSON body"}), 400
 
    if "email_text" in data:
        email_text = data.get("email_text", "").strip()
        if not email_text:
            return jsonify({"error": "email_text cannot be empty"}), 400
        try:
            return jsonify(predict_email(email_text)), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
 
    elif "url" in data:
        url = data.get("url", "").strip()
        if not url:
            return jsonify({"error": "URL cannot be empty"}), 400
        try:
            return jsonify(predict_url(url)), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
 
    else:
        return jsonify({"error": "Provide 'url' or 'email_text' in request body"}), 400
 
 
if __name__ == "__main__":
    app.run(debug=True)
 