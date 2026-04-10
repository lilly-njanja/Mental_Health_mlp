import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import db_utils
from datetime import datetime

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MindGuard · Mental Health Risk",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300;1,600&display=swap');

html, body, [class*="css"] { font-family: 'Poppins', sans-serif; color: #1a1a2e; }
.stApp { background: #f0f4fa; }

[data-testid="stSidebar"] { background: #1a1a2e !important; border-right: none; }
[data-testid="stSidebar"] * { color: #e8e4da !important; }
[data-testid="stSidebar"] .stRadio label { color: #e8e4da !important; font-size: 0.95rem !important; font-weight: 500 !important; padding: 6px 0 !important; }
[data-testid="stSidebar"] .stRadio > div { gap: 4px; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.12) !important; }

#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }

.page-title { font-family: 'Poppins', sans-serif; font-size: 2.6rem; font-weight: 600; color: #1a1a2e; margin-bottom: 0.1rem; line-height: 1.15; }
.page-subtitle { font-size: 1rem; color: #5a5a72; font-weight: 300; margin-bottom: 2rem; }

.stat-card { background: #ffffff; border-radius: 16px; padding: 1.6rem 1.8rem; box-shadow: 0 2px 16px rgba(0,0,0,0.06); border-left: 4px solid #6c63ff; margin-bottom: 1rem; }
.stat-card .stat-number { font-family: 'Poppins', sans-serif; font-size: 2.2rem; color: #1a1a2e; font-weight: 600; }
.stat-card .stat-label { font-size: 0.82rem; color: #7a7a8c; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 500; }

.feature-card { background: #ffffff; border-radius: 16px; padding: 1.5rem 1.8rem; box-shadow: 0 2px 16px rgba(0,0,0,0.06); height: 100%; margin-bottom: 1rem; }
.feature-card .feat-icon { font-size: 1.8rem; margin-bottom: 0.6rem; }
.feature-card .feat-title { font-family: 'Poppins', sans-serif; font-size: 1.05rem; font-weight: 600; color: #1a1a2e; margin-bottom: 0.3rem; }
.feature-card .feat-desc { font-size: 0.88rem; color: #5a5a72; line-height: 1.5; }

.form-card { background: #ffffff; border-radius: 20px; padding: 2rem 2.2rem; box-shadow: 0 2px 20px rgba(0,0,0,0.07); margin-bottom: 1.5rem; }
.form-section-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.13em; text-transform: uppercase; color: #6c63ff; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid #ede9e0; }

label { color: #2a2a3e !important; font-size: 0.88rem !important; font-weight: 500 !important; }
p, li, span { color: #1a1a2e; }

.stButton > button { background: #6c63ff !important; color: #ffffff !important; border: none !important; border-radius: 12px !important; padding: 0.75rem 2.2rem !important; font-size: 0.95rem !important; font-weight: 600 !important; letter-spacing: 0.04em !important; width: 100% !important; transition: background 0.2s ease !important; }
.stButton > button p, .stButton > button span { color: #ffffff !important; }
.stButton > button:hover { background: #1a1a2e !important; }

.result-wrap { border-radius: 18px; padding: 2rem 2.4rem; margin-top: 1.5rem; text-align: center; }
.result-low      { background: #e8f5e9; border: 2px solid #43a047; }
.result-moderate { background: #fff8e1; border: 2px solid #f9a825; }
.result-high     { background: #fce4ec; border: 2px solid #e53935; }
.result-badge { font-family: 'Poppins', sans-serif; font-size: 2rem; font-weight: 600; margin-bottom: 0.3rem; }
.result-low      .result-badge { color: #2e7d32; }
.result-moderate .result-badge { color: #f57f17; }
.result-high     .result-badge { color: #c62828; }
.result-desc { font-size: 0.95rem; color: #3a3a4e; margin-top: 0.4rem; }

.prob-row { margin-bottom: 10px; }
.prob-row-header { display: flex; justify-content: space-between; font-size: 0.82rem; color: #5a5a72; margin-bottom: 4px; font-weight: 500; }
.prob-track { background: #ede9e0; border-radius: 8px; height: 12px; overflow: hidden; }
.prob-fill-low  { background: linear-gradient(90deg,#43a047,#66bb6a); height: 100%; border-radius: 8px; }
.prob-fill-mod  { background: linear-gradient(90deg,#f9a825,#ffca28); height: 100%; border-radius: 8px; }
.prob-fill-high { background: linear-gradient(90deg,#e53935,#ef5350); height: 100%; border-radius: 8px; }

.about-card { background: #ffffff; border-radius: 16px; padding: 1.8rem 2rem; box-shadow: 0 2px 16px rgba(0,0,0,0.06); margin-bottom: 1.2rem; }
.about-card h4 { font-family: 'Poppins', sans-serif; color: #1a1a2e; font-size: 1.1rem; margin-bottom: 0.6rem; }
.about-card p, .about-card li { color: #3a3a4e; font-size: 0.9rem; line-height: 1.7; }

.disclaimer { background: #eef2ff; border-left: 4px solid #6c63ff; border-radius: 0 12px 12px 0; padding: 1rem 1.2rem; font-size: 0.85rem; color: #3a3a4e; margin-top: 1.5rem; }
.rec-card { background: #fff; border-radius: 14px; padding: 1rem 1.3rem; box-shadow: 0 2px 10px rgba(0,0,0,0.05); margin-bottom: 0.8rem; border-left: 4px solid #6c63ff; font-size: 0.88rem; color: #2a2a3e; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state["page"] = "🏠 Home"
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "is_admin" not in st.session_state:
    st.session_state["is_admin"] = False

# ═══════════════════════════════════════════════════════════
# TOP NAVIGATION
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;'>
    <div style='display: flex; align-items: center; gap: 0.8rem;'>
        <div style='font-size:2rem; animation: float 3s ease-in-out infinite;'>🧠</div>
        <div>
            <div style='font-family:"Poppins",sans-serif;font-size:1.4rem;font-weight:700;color:#1a1a2e;line-height:1.2;'>MindGuard</div>
            <div style='font-size:0.7rem;color:#7a7a8c;letter-spacing:0.05em;text-transform:uppercase;'>Mental Health Risk</div>
        </div>
    </div>
    <div style='font-size:0.95rem; color:#6c63ff; font-weight: 500;'>
""", unsafe_allow_html=True)
if st.session_state["logged_in"]:
    st.markdown(f"Welcome, <b style='color:#1a1a2e;'>{st.session_state['username']}</b></div></div>", unsafe_allow_html=True)
else:
    st.markdown("</div></div>", unsafe_allow_html=True)

if st.session_state["logged_in"]:
    if st.session_state["is_admin"]:
        pages = ["🛡️ Admin Dashboard", "🔓 Logout"]
    else:
        pages = ["🏠 Home", "🔮 Prediction", "📜 My History", "📖 About", "🔓 Logout"]
else:
    pages = ["🔑 Login / Register"]
    if st.session_state["page"] not in pages:
        st.session_state["page"] = "🔑 Login / Register"

if len(pages) > 1:
    nav_cols = st.columns(len(pages))
    for idx, page_name in enumerate(pages):
        with nav_cols[idx]:
            btn_type = "primary" if st.session_state["page"] == page_name else "secondary"
            if st.button(page_name, use_container_width=True, type=btn_type, key=f"nav_top_{idx}"):
                st.session_state["page"] = page_name
                st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)

if st.session_state["page"] == "🔓 Logout":
    st.session_state["logged_in"] = False
    st.session_state["username"] = None
    st.session_state["user_id"] = None
    st.session_state["is_admin"] = False
    st.session_state["page"] = "🏠 Home"
    st.rerun()

# ═══════════════════════════════════════════════════════════
# LOAD MODEL BUNDLE + COMPARISON MODELS
# ═══════════════════════════════════════════════════════════
@st.cache_resource
def load_all_models():
    bundle = joblib.load("mental_health_model_bundle.pkl")
    extras = {}
    for name, fname in [("Logistic Regression", "lr_model.pkl"),
                        ("Random Forest",        "rf_model.pkl"),
                        ("MLP",                  "mlp_model.pkl")]:
        if os.path.exists(fname):
            try:
                extras[name] = joblib.load(fname)
            except Exception:
                pass
    return bundle, extras

try:
    bundle, extra_models    = load_all_models()
    model                   = bundle["model"]
    scaler                  = bundle["scaler"]
    feature_columns         = bundle["feature_columns"]
    categorical_columns     = bundle["categorical_columns"]
    model_loaded            = True
except Exception as e:
    model_loaded = False
    load_error   = str(e)

# ═══════════════════════════════════════════════════════════
# FEATURE CONFIG  (covers every column the dataset may have)
# ═══════════════════════════════════════════════════════════
KNOWN_NUMERIC = {
    "Age":                    {"label": "Age",                        "min": 18,    "max": 80,     "default": 30,    "step": 1.0},
    "Sleep_Hours":            {"label": "Sleep Hours (per night)",    "min": 0.0,   "max": 12.0,   "default": 7.0,   "step": 0.5},
    "Physical_Activity_Hrs":  {"label": "Physical Activity (hrs/wk)", "min": 0.0,   "max": 14.0,   "default": 3.0,   "step": 0.5},
    "Social_Support_Score":   {"label": "Social Support Score",       "min": 0,     "max": 10,     "default": 5,     "step": 1.0},
    "Financial_Stress":       {"label": "Financial Stress",           "min": 0,     "max": 10,     "default": 5,     "step": 1.0},
    "Work_Stress":            {"label": "Work Stress",                "min": 0,     "max": 10,     "default": 5,     "step": 1.0},
    "Self_Esteem_Score":      {"label": "Self-Esteem Score",          "min": 0,     "max": 10,     "default": 5,     "step": 1.0},
    "Life_Satisfaction_Score":{"label": "Life Satisfaction Score",    "min": 0,     "max": 10,     "default": 5,     "step": 1.0},
    "Loneliness_Score":       {"label": "Loneliness Score",           "min": 0,     "max": 10,     "default": 5,     "step": 1.0},
    "Physical_Health_Score":  {"label": "Physical Health Score",      "min": 0,     "max": 10,     "default": 5,     "step": 1.0},
    "Diet_Quality":           {"label": "Diet Quality Score",         "min": 0,     "max": 10,     "default": 5,     "step": 1.0},
    "Screen_Time_Hrs":        {"label": "Screen Time (hrs/day)",      "min": 0.0,   "max": 16.0,   "default": 4.0,   "step": 0.5},
    "Air_Quality_Index":      {"label": "Air Quality Index",          "min": 0,     "max": 500,    "default": 100,   "step": 1.0},
    "Hours_Worked_Per_Week":  {"label": "Hours Worked Per Week",      "min": 0,     "max": 80,     "default": 40,    "step": 1.0},
    "Number_of_Children":     {"label": "Number of Children",         "min": 0,     "max": 10,     "default": 0,     "step": 1.0},
    "Income":                 {"label": "Annual Income (USD)",        "min": 0,     "max": 200000, "default": 40000, "step": 1000.0},
}

KNOWN_CATEGORICAL = {
    "Gender":            {"label": "Gender",                          "options": ["Male","Female","Non-Binary","Other"],               "map": {"Male":0,"Female":1,"Non-Binary":2,"Other":3}},
    "Education_Level":   {"label": "Education Level",                 "options": ["High School","Bachelor's","Master's","PhD","Other"],"map": {"High School":0,"Bachelor's":1,"Master's":2,"PhD":3,"Other":4}},
    "Employment_Status": {"label": "Employment Status",               "options": ["Employed","Unemployed","Student","Retired"],        "map": {"Unemployed":0,"Retired":1,"Employed":2,"Student":3}},
    "Medication_Use":    {"label": "Medication Use",                  "options": ["None","Occasional","Regular"],                      "map": {"None":0,"Occasional":1,"Regular":2}},
    "Substance_Use":     {"label": "Substance Use",                   "options": ["None","Occasional","Frequent"],                     "map": {"None":0,"Occasional":1,"Frequent":2}},
    "Family_History_Mental_Illness": {"label": "Family History of Mental Illness","options": ["No","Yes"],                             "map": {"No":0,"Yes":1}},
    "Chronic_Illnesses": {"label": "Chronic Illnesses",               "options": ["No","Yes"],                                         "map": {"No":0,"Yes":1}},
    "Therapy":           {"label": "Currently in Therapy",            "options": ["No","Yes"],                                         "map": {"No":0,"Yes":1}},
    "Meditation":        {"label": "Practices Meditation",            "options": ["No","Yes"],                                         "map": {"No":0,"Yes":1}},
    "Marital_Status":    {"label": "Marital Status",                  "options": ["Single","Married","Divorced","Widowed"],            "map": {"Single":0,"Married":1,"Divorced":2,"Widowed":3}},
    "Smoking_Status":    {"label": "Smoking Status",                  "options": ["Never","Former","Current"],                        "map": {"Never":0,"Former":1,"Current":2}},
    "Alcohol_Consumption":{"label": "Alcohol Consumption",            "options": ["None","Occasional","Regular","Heavy"],             "map": {"None":0,"Occasional":1,"Regular":2,"Heavy":3}},
    "Urban_Rural":       {"label": "Urban / Rural",                   "options": ["Urban","Rural"],                                    "map": {"Urban":0,"Rural":1}},
    "Sleep_Quality":     {"label": "Sleep Quality",                   "options": ["Poor","Fair","Good","Excellent"],                  "map": {"Poor":0,"Fair":1,"Good":2,"Excellent":3}},
    "Exercise_Frequency":{"label": "Exercise Frequency",              "options": ["Never","Rarely","Sometimes","Often","Daily"],      "map": {"Never":0,"Rarely":1,"Sometimes":2,"Often":3,"Daily":4}},
}

# Numeric features the scaler was fitted on
SCALER_FEATURES = [
    "Age","Sleep_Hours","Physical_Activity_Hrs","Social_Support_Score",
    "Financial_Stress","Work_Stress","Self_Esteem_Score",
    "Life_Satisfaction_Score","Loneliness_Score",
    "Physical_Health_Score","Diet_Quality","Screen_Time_Hrs",
    "Air_Quality_Index","Hours_Worked_Per_Week","Number_of_Children","Income",
]

# Form section groupings
SECTION_NUMERIC = {
    "👤 Demographics":  ["Age"],
    "🌿 Lifestyle":     ["Sleep_Hours","Physical_Activity_Hrs","Screen_Time_Hrs","Diet_Quality","Hours_Worked_Per_Week"],
    "💬 Psychosocial":  ["Social_Support_Score","Self_Esteem_Score","Life_Satisfaction_Score","Loneliness_Score","Financial_Stress","Work_Stress"],
    "🏥 Health":        ["Physical_Health_Score","Air_Quality_Index","Number_of_Children","Income"],
}
SECTION_CATEGORICAL = {
    "👤 Demographics":  ["Gender","Education_Level","Employment_Status","Marital_Status","Urban_Rural"],
    "🌿 Lifestyle":     ["Medication_Use","Substance_Use","Smoking_Status","Alcohol_Consumption","Exercise_Frequency","Sleep_Quality"],
    "💬 Psychosocial":  [],
    "🏥 Health":        ["Family_History_Mental_Illness","Chronic_Illnesses","Therapy","Meditation"],
}

# ═══════════════════════════════════════════════════════════
# PDF GENERATOR
# ═══════════════════════════════════════════════════════════
def generate_pdf(inputs_display, predicted_label, probabilities, recommendations, comp_results=None):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        from reportlab.lib.units import inch
    except ImportError:
        return None

    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=letter,
                               rightMargin=0.75*inch, leftMargin=0.75*inch,
                               topMargin=0.75*inch,   bottomMargin=0.75*inch)

    NAVY   = colors.HexColor("#1a1a2e")
    INDIGO = colors.HexColor("#6c63ff")
    GREEN  = colors.HexColor("#2e7d32")
    AMBER  = colors.HexColor("#f57f17")
    RED    = colors.HexColor("#c62828")
    LIGHT  = colors.HexColor("#f4f1eb")

    risk_color = {"Low": GREEN, "Moderate": AMBER, "High": RED}.get(predicted_label, NAVY)

    title_style = ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=20, textColor=NAVY,  spaceAfter=4)
    sub_style   = ParagraphStyle("sub",   fontName="Helvetica",      fontSize=10, textColor=colors.HexColor("#5a5a72"), spaceAfter=12)
    h2_style    = ParagraphStyle("h2",    fontName="Helvetica-Bold", fontSize=13, textColor=NAVY,  spaceBefore=14, spaceAfter=6)
    body_style  = ParagraphStyle("body",  fontName="Helvetica",      fontSize=9,  textColor=colors.HexColor("#3a3a4e"), leading=14)
    risk_style  = ParagraphStyle("risk",  fontName="Helvetica-Bold", fontSize=22, textColor=risk_color, alignment=1, spaceBefore=8, spaceAfter=4)
    small_style = ParagraphStyle("small", fontName="Helvetica",      fontSize=8,  textColor=colors.HexColor("#7a7a8c"), leading=12)

    story = []
    story.append(Paragraph("🧠 MindGuard", title_style))
    story.append(Paragraph("Mental Health Risk Assessment Report", sub_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y  %H:%M')}", small_style))
    story.append(HRFlowable(width="100%", thickness=1, color=INDIGO, spaceAfter=14))

    # Primary result
    story.append(Paragraph("Assessment Result", h2_style))
    emoji = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}.get(predicted_label, "")
    story.append(Paragraph(f"{emoji}  {predicted_label} Risk", risk_style))
    story.append(Spacer(1, 8))

    # XGBoost probability table
    prob_data = [["Risk Level", "Probability"]]
    for lbl, prob in zip(["Low","Moderate","High"], probabilities):
        prob_data.append([lbl, f"{prob*100:.1f}%"])
    t = Table(prob_data, colWidths=[2.5*inch, 2*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  NAVY),
        ("TEXTCOLOR",     (0,0),(-1,0),  colors.white),
        ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 9),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, LIGHT]),
        ("GRID",          (0,0),(-1,-1), 0.4, colors.HexColor("#ede9e0")),
        ("ALIGN",         (1,0),(1,-1),  "CENTER"),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 14))

    # Model comparison table (if available)
    if comp_results:
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#ede9e0"), spaceAfter=8))
        story.append(Paragraph("Model Comparison", h2_style))
        cmp_data = [["Model", "Prediction", "Low %", "Moderate %", "High %"]]
        for mname, res in comp_results.items():
            cmp_data.append([
                mname,
                res["pred"],
                f"{res['probs'][0]*100:.1f}%",
                f"{res['probs'][1]*100:.1f}%",
                f"{res['probs'][2]*100:.1f}%",
            ])
        t2 = Table(cmp_data, colWidths=[1.7*inch, 1.1*inch, 1*inch, 1.1*inch, 1*inch])
        t2.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,0),  NAVY),
            ("TEXTCOLOR",     (0,0),(-1,0),  colors.white),
            ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, LIGHT]),
            ("GRID",          (0,0),(-1,-1), 0.4, colors.HexColor("#ede9e0")),
            ("ALIGN",         (1,0),(-1,-1), "CENTER"),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0),(-1,-1), 4),
            ("BOTTOMPADDING", (0,0),(-1,-1), 4),
        ]))
        story.append(t2)
        story.append(Spacer(1, 14))

    # Input summary
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#ede9e0"), spaceAfter=8))
    story.append(Paragraph("Input Summary", h2_style))
    input_rows = [["Feature", "Value"]]
    for k, v in inputs_display.items():
        input_rows.append([k, str(v)])
    t3 = Table(input_rows, colWidths=[3*inch, 3*inch])
    t3.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  NAVY),
        ("TEXTCOLOR",     (0,0),(-1,0),  colors.white),
        ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, LIGHT]),
        ("GRID",          (0,0),(-1,-1), 0.4, colors.HexColor("#ede9e0")),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),(-1,-1), 4),
        ("BOTTOMPADDING", (0,0),(-1,-1), 4),
    ]))
    story.append(t3)
    story.append(Spacer(1, 14))

    # Recommendations
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#ede9e0"), spaceAfter=8))
    story.append(Paragraph("Personalised Recommendations", h2_style))
    for rec in recommendations:
        story.append(Paragraph(f"• {rec}", body_style))
        story.append(Spacer(1, 3))

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#ede9e0"), spaceAfter=6))
    story.append(Paragraph(
        "⚠️ Disclaimer: This report is generated by a machine learning model for informational purposes only. "
        "It is not a substitute for professional mental health evaluation, diagnosis, or treatment. "
        "Please consult a qualified healthcare provider.",
        small_style
    ))

    doc.build(story)
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════
if st.session_state["page"] == "🏠 Home":

    st.markdown('<h1 class="page-title">Mental Health<br>Risk Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">An intelligent, AI-powered screening tool to assess psychological well-being across key life dimensions.</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, (num, label, color) in zip([c1,c2,c3,c4], [
        ("AI",          "Powered Analysis",       "#6c63ff"),
        ("3",           "Risk Categories",        "#e53935"),
        ("15+",         "Indicators Analyzed",    "#43a047"),
        ("100%",        "Private & Secure",       "#f9a825"),
    ]):
        with col:
            st.markdown(f"""
            <div class="stat-card" style="border-left-color:{color};">
                <div class="stat-number">{num}</div>
                <div class="stat-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="form-section-label">What this app covers</p>', unsafe_allow_html=True)

    features = [
        ("👤","Demographics",        "Age, gender, education, employment, marital status and region."),
        ("🌿","Lifestyle",           "Sleep, physical activity, screen time, diet quality, substance use and smoking."),
        ("💬","Psychosocial Scores", "Social support, self-esteem, loneliness, life satisfaction, financial and work stress."),
        ("🏥","Health Background",   "Physical health, chronic illness, family history, therapy and meditation."),
        ("🤖","Model Comparison",    "Compare XGBoost, Logistic Regression, Random Forest and MLP predictions side-by-side."),
        ("📄","PDF Report",          "Download a full personalised report including inputs, results and recommendations."),
    ]
    cols_cycle = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols_cycle[i % 3]:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feat-icon">{icon}</div>
                <div class="feat-title">{title}</div>
                <div class="feat-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
    <strong>How to use:</strong> Navigate to <strong>Prediction</strong> in the sidebar, fill in your details,
    and click <em>Assess My Risk</em> to get your result, interactive charts, a model comparison, and a downloadable PDF report.
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Continue to Prediction ➡️", key="nav_home_pred", use_container_width=True):
        st.session_state["page"] = "🔮 Prediction"
        st.rerun()


# ═══════════════════════════════════════════════════════════
# PAGE: PREDICTION
# ═══════════════════════════════════════════════════════════
elif st.session_state["page"] == "🔮 Prediction":

    st.markdown('<h1 class="page-title">Risk Assessment</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Fill in all sections below for your personalised mental health risk score.</p>', unsafe_allow_html=True)

    if not model_loaded:
        st.error(f"Could not load model bundle: {load_error}\n\nEnsure `mental_health_model_bundle.pkl` is in the same directory as `app.py`.")
        st.stop()

    # ── Determine which features the model actually needs ──
    needed_numeric     = [f for f in feature_columns if f in KNOWN_NUMERIC]
    needed_categorical = [f for f in feature_columns if f in KNOWN_CATEGORICAL]
    unknown_features   = [f for f in feature_columns if f not in KNOWN_NUMERIC and f not in KNOWN_CATEGORICAL]

    input_values   = {}   # encoded integer / float values for model
    display_values = {}   # human-readable strings for PDF

    # ── Render form sections ──
    for section, num_keys in SECTION_NUMERIC.items():
        active_num = [k for k in num_keys if k in needed_numeric]
        active_cat = [k for k in SECTION_CATEGORICAL.get(section, []) if k in needed_categorical]
        if not active_num and not active_cat:
            continue

        st.markdown('<div class="form-card">', unsafe_allow_html=True)
        st.markdown(f'<p class="form-section-label">{section}</p>', unsafe_allow_html=True)

        all_fields = active_num + active_cat
        mid = (len(all_fields) + 1) // 2
        col1, col2 = st.columns(2)

        for i, key in enumerate(all_fields):
            target_col = col1 if i < mid else col2
            with target_col:
                if key in KNOWN_NUMERIC:
                    cfg  = KNOWN_NUMERIC[key]
                    step = cfg["step"]
                    mn, mx, dv = cfg["min"], cfg["max"], cfg["default"]
                    if step == 1.0 and isinstance(mn, int):
                        val = st.slider(cfg["label"], int(mn), int(mx), int(dv))
                    else:
                        val = st.slider(cfg["label"], float(mn), float(mx), float(dv), float(step))
                    input_values[key]              = val
                    display_values[cfg["label"]]   = val
                else:
                    cfg    = KNOWN_CATEGORICAL[key]
                    choice = st.selectbox(cfg["label"], cfg["options"])
                    input_values[key]              = cfg["map"][choice]
                    display_values[cfg["label"]]   = choice

        st.markdown('</div>', unsafe_allow_html=True)

    # Default 0 for any unknown features
    for key in unknown_features:
        input_values[key] = 0

    # ── Predict button ──
    if st.button("🔍 Assess My Risk"):

        # Build and preprocess input for XGBoost
        raw_row  = {k: input_values.get(k, 0) for k in feature_columns}
        input_df = pd.DataFrame([raw_row])[feature_columns]

        # Use exactly the columns the scaler was fitted on
        if hasattr(scaler, "feature_names_in_"):
            scale_cols = [c for c in scaler.feature_names_in_ if c in input_df.columns]
        else:
            scale_cols = [c for c in SCALER_FEATURES if c in input_df.columns]
        if scale_cols:
            input_df[scale_cols] = scaler.transform(input_df[scale_cols])

        for col in categorical_columns:
            if col in input_df.columns:
                input_df[col] = input_df[col].astype("category")

        # XGBoost prediction
        prediction      = model.predict(input_df)[0]
        probabilities   = model.predict_proba(input_df)[0]
        label_map       = {0: "Low", 1: "Moderate", 2: "High"}
        predicted_label = label_map[int(prediction)]

        # Recommendations
        if predicted_label == "Low":
            css, emoji, desc = "result-low", "🟢", "Your indicators suggest a generally healthy mental state. Keep nurturing your positive habits."
            recommendations = [
                "Maintain your current sleep schedule of 7–9 hours.",
                "Continue regular physical activity — it's one of the best mental health protectors.",
                "Stay socially connected with friends and family.",
                "Schedule an annual mental health check-in as a preventive measure.",
                "Practice gratitude journaling to reinforce positive thinking.",
            ]
        elif predicted_label == "Moderate":
            css, emoji, desc = "result-moderate", "🟡", "Some risk factors detected. Mindfulness, social connection, and professional support could help."
            recommendations = [
                "Prioritise sleep — aim for 7–9 consistent hours per night.",
                "Add 20–30 minutes of moderate exercise at least 3× per week.",
                "Consider speaking with a counsellor or therapist.",
                "Limit alcohol and substance use, which can worsen anxiety.",
                "Try mindfulness or meditation apps (e.g. Headspace, Calm) daily.",
                "Reach out to trusted friends or a support group.",
            ]
        else:
            css, emoji, desc = "result-high", "🔴", "High risk indicators detected. We strongly encourage reaching out to a mental health professional soon."
            recommendations = [
                "Consult a mental health professional (psychologist or psychiatrist) as soon as possible.",
                "Contact a mental health helpline if you need immediate support.",
                "Avoid self-medicating with alcohol or substances.",
                "Share your feelings with a trusted person — isolation worsens risk.",
                "Ask your doctor about evidence-based therapies (e.g. CBT).",
                "Monitor your mood daily using a journal or app.",
            ]

        # Save to DB
        if st.session_state.get("user_id"):
            db_utils.save_assessment(
                user_id=st.session_state["user_id"],
                predicted_risk=predicted_label,
                probabilities=list(float(p) for p in probabilities),
                inputs_dict=input_values
            )

        # ── PRIMARY RESULT ──────────────────────────────────
        st.markdown("---")
        st.markdown("### 📋 Your Results")

        res_col, prob_col = st.columns(2)

        with res_col:
            st.markdown(f"""
            <div class="result-wrap {css}">
                <div style="font-size:2.4rem;">{emoji}</div>
                <div class="result-badge">{predicted_label} Risk</div>
                <p class="result-desc">{desc}</p>
            </div>""", unsafe_allow_html=True)

        with prob_col:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="form-section-label">📊 XGBoost Probability Breakdown</p>', unsafe_allow_html=True)
            for lbl, fill_class, prob in zip(
                ["Low","Moderate","High"],
                ["prob-fill-low","prob-fill-mod","prob-fill-high"],
                probabilities
            ):
                pct = prob * 100
                st.markdown(f"""
                <div class="prob-row">
                    <div class="prob-row-header"><span>{lbl}</span><span>{pct:.1f}%</span></div>
                    <div class="prob-track"><div class="{fill_class}" style="width:{pct}%"></div></div>
                </div>""", unsafe_allow_html=True)

        # ── VISUALISATIONS ──────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 Visualisations")

        try:
            import plotly.graph_objects as go

            v1, v2 = st.columns(2)

            with v1:
                high_prob   = float(probabilities[2]) * 100
                gauge_color = "#43a047" if predicted_label=="Low" else ("#f9a825" if predicted_label=="Moderate" else "#e53935")
                fig_gauge   = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=high_prob,
                    title={"text": "High-Risk Probability (%)", "font": {"size": 13}},
                    gauge={
                        "axis": {"range": [0,100]},
                        "bar":  {"color": gauge_color},
                        "steps": [
                            {"range": [0, 33],  "color": "#e8f5e9"},
                            {"range": [33, 66], "color": "#fff8e1"},
                            {"range": [66,100], "color": "#fce4ec"},
                        ],
                        "threshold": {"line": {"color":"#1a1a2e","width":3}, "thickness":0.75, "value":high_prob}
                    },
                    number={"suffix":"%","font":{"size":28}},
                ))
                fig_gauge.update_layout(height=280, margin=dict(t=50,b=10,l=20,r=20), paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_gauge, use_container_width=True)

            with v2:
                fig_donut = go.Figure(go.Pie(
                    labels=["Low","Moderate","High"],
                    values=[float(p)*100 for p in probabilities],
                    hole=0.55,
                    marker_colors=["#43a047","#f9a825","#e53935"],
                    textinfo="label+percent",
                    textfont_size=12,
                ))
                fig_donut.update_layout(
                    title="Risk Distribution (XGBoost)",
                    height=280,
                    margin=dict(t=50,b=10,l=10,r=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                )
                st.plotly_chart(fig_donut, use_container_width=True)

            # Psychosocial bar chart
            radar_keys = [k for k in ["Social_Support_Score","Self_Esteem_Score","Life_Satisfaction_Score",
                                       "Loneliness_Score","Financial_Stress","Work_Stress"]
                          if k in input_values]
            if radar_keys:
                radar_labels = [k.replace("_"," ") for k in radar_keys]
                radar_vals   = [input_values[k] for k in radar_keys]
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=radar_labels,
                    y=radar_vals,
                    marker_color=["#43a047" if v<=4 else ("#f9a825" if v<=7 else "#e53935") for v in radar_vals],
                    text=[str(v) for v in radar_vals],
                    textposition="outside",
                ))
                fig_bar.update_layout(
                    title="Your Psychosocial Scores (0–10)",
                    yaxis=dict(range=[0,12]),
                    height=300,
                    margin=dict(t=50,b=40,l=20,r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Poppins, sans-serif", size=11),
                )
                fig_bar.update_xaxes(showgrid=False)
                fig_bar.update_yaxes(gridcolor="#ede9e0")
                st.plotly_chart(fig_bar, use_container_width=True)

        except ImportError:
            st.info("Install `plotly` for interactive charts: `pip install plotly`")

        # ── MODEL COMPARISON ────────────────────────────────
        st.markdown("---")
        st.markdown("### 🤖 Model Comparison")
        st.markdown(
            '<p style="color:#5a5a72;font-size:0.92rem;margin-bottom:1rem;">'
            'Your inputs are run through all four trained models. '
            '<strong>XGBoost ★</strong> is the primary model — the others are shown for comparison.'
            '</p>',
            unsafe_allow_html=True
        )

        # Build a numeric-only (int) dataframe for sklearn models
        input_df_sk = pd.DataFrame([raw_row])[feature_columns].copy()
        # Use exactly the columns the scaler was fitted on
        if hasattr(scaler, "feature_names_in_"):
            scale_cols_sk = [c for c in scaler.feature_names_in_ if c in input_df_sk.columns]
        else:
            scale_cols_sk = [c for c in SCALER_FEATURES if c in input_df_sk.columns]
        if scale_cols_sk:
            input_df_sk[scale_cols_sk] = scaler.transform(input_df_sk[scale_cols_sk])
        for col in categorical_columns:
            if col in input_df_sk.columns:
                input_df_sk[col] = input_df_sk[col].astype(int)

        label_map_sk  = {0:"Low", 1:"Moderate", 2:"High"}
        risk_colors   = {"Low":"#43a047","Moderate":"#f9a825","High":"#e53935"}

        # Collect predictions from all available models
        comp_results = {}

        # XGBoost is always present
        comp_results["XGBoost ★"] = {"pred": predicted_label, "probs": probabilities}

        # Saved comparison models (lr, rf, mlp)
        for mname, m in extra_models.items():
            try:
                probs = m.predict_proba(input_df_sk)[0]
                pred  = label_map_sk[int(m.predict(input_df_sk)[0])]
                if len(probs) == 3:
                    comp_results[mname] = {"pred": pred, "probs": probs}
            except Exception:
                pass

        have_all_four = len(comp_results) == 4

        if not have_all_four:
            missing = {"Logistic Regression","Random Forest","MLP"} - set(extra_models.keys())
            st.warning(
                f"⚠️ Comparison model(s) not found: **{', '.join(missing)}**. "
                "Place `lr_model.pkl`, `rf_model.pkl`, and `mlp_model.pkl` "
                "in the same folder as `app.py` to enable full comparison."
            )

        try:
            import plotly.graph_objects as go

            # ── Summary cards ──
            card_cols = st.columns(len(comp_results))
            for (mname, res), col in zip(comp_results.items(), card_cols):
                risk_col = risk_colors.get(res["pred"], "#6c63ff")
                with col:
                    st.markdown(f"""
                    <div style="background:#fff;border-radius:14px;padding:1.2rem 1rem;
                                box-shadow:0 2px 12px rgba(0,0,0,0.07);text-align:center;
                                border-top:4px solid {risk_col};margin-bottom:0.5rem;">
                        <div style="font-size:0.7rem;font-weight:600;letter-spacing:0.1em;
                                    text-transform:uppercase;color:#7a7a8c;margin-bottom:0.4rem;">{mname}</div>
                        <div style="font-family:'Poppins',sans-serif;font-size:1.4rem;
                                    font-weight:600;color:{risk_col};">{res["pred"]}</div>
                        <div style="font-size:0.78rem;color:#9a9aac;margin-top:0.2rem;">
                            High: {res["probs"][2]*100:.1f}%</div>
                    </div>""", unsafe_allow_html=True)

            # ── Grouped bar chart ──
            model_names = list(comp_results.keys())
            low_probs   = [comp_results[m]["probs"][0]*100 for m in model_names]
            mod_probs   = [comp_results[m]["probs"][1]*100 for m in model_names]
            high_probs  = [comp_results[m]["probs"][2]*100 for m in model_names]

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(name="Low Risk",      x=model_names, y=low_probs,  marker_color="#43a047",
                                     text=[f"{v:.1f}%" for v in low_probs],  textposition="outside"))
            fig_cmp.add_trace(go.Bar(name="Moderate Risk", x=model_names, y=mod_probs,  marker_color="#f9a825",
                                     text=[f"{v:.1f}%" for v in mod_probs],  textposition="outside"))
            fig_cmp.add_trace(go.Bar(name="High Risk",     x=model_names, y=high_probs, marker_color="#e53935",
                                     text=[f"{v:.1f}%" for v in high_probs], textposition="outside"))
            fig_cmp.update_layout(
                barmode="group",
                title="Probability Comparison Across All Models",
                yaxis=dict(range=[0,115], title="Probability (%)"),
                xaxis_title="Model",
                height=380,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=60,b=40,l=20,r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Poppins, sans-serif", size=11),
            )
            fig_cmp.update_xaxes(showgrid=False)
            fig_cmp.update_yaxes(gridcolor="#ede9e0")
            st.plotly_chart(fig_cmp, use_container_width=True)

            # ── Heatmap ──
            z_vals = [[comp_results[m]["probs"][i]*100 for m in model_names] for i in range(3)]
            fig_heat = go.Figure(go.Heatmap(
                z=z_vals,
                x=model_names,
                y=["Low","Moderate","High"],
                colorscale=[[0,"#e8f5e9"],[0.5,"#fff8e1"],[1,"#fce4ec"]],
                text=[[f"{v:.1f}%" for v in row] for row in z_vals],
                texttemplate="%{text}",
                textfont={"size":12,"color":"#1a1a2e"},
                showscale=False,
                zmin=0, zmax=100,
            ))
            fig_heat.update_layout(
                title="Risk Probability Heatmap (All Models)",
                height=240,
                margin=dict(t=50,b=20,l=80,r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Poppins, sans-serif", size=11),
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            # ── Agreement indicator ──
            all_preds   = [res["pred"] for res in comp_results.values()]
            n_agree     = all_preds.count(predicted_label)
            agree_pct   = n_agree / len(all_preds) * 100
            agree_color = "#43a047" if agree_pct >= 75 else ("#f9a825" if agree_pct >= 50 else "#e53935")
            st.markdown(f"""
            <div style="background:#fff;border-radius:14px;padding:1rem 1.4rem;
                        box-shadow:0 2px 12px rgba(0,0,0,0.06);display:flex;
                        align-items:center;gap:1rem;margin-top:0.5rem;">
                <div style="font-size:2rem;">🤝</div>
                <div>
                    <div style="font-weight:600;font-size:0.95rem;color:#1a1a2e;">
                        Model Agreement: <span style="color:{agree_color};">
                        {n_agree}/{len(all_preds)} models predict <em>{predicted_label}</em> Risk
                        </span>
                    </div>
                    <div style="font-size:0.82rem;color:#7a7a8c;margin-top:0.2rem;">
                        {agree_pct:.0f}% consensus · XGBoost ★ is the primary model used for your result
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        except ImportError:
            st.info("Install `plotly` for comparison charts: `pip install plotly`")

        # ── RECOMMENDATIONS ─────────────────────────────────
        st.markdown("---")
        st.markdown("### 💡 Personalised Recommendations")
        rec_cols = st.columns(2)
        for i, rec in enumerate(recommendations):
            with rec_cols[i % 2]:
                st.markdown(f'<div class="rec-card">💬 {rec}</div>', unsafe_allow_html=True)

        # ── PDF DOWNLOAD ─────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📄 Download Your Report")
        pdf_buf = generate_pdf(display_values, predicted_label, probabilities, recommendations,
                               comp_results if len(comp_results) > 1 else None)
        if pdf_buf:
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_buf,
                file_name=f"mindguard_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.info("Install `reportlab` to enable PDF download: `pip install reportlab`")

        st.markdown("""
        <div class="disclaimer">
        ⚠️ <strong>Disclaimer:</strong> This assessment is for informational purposes only and is
        <strong>not</strong> a substitute for professional mental health evaluation, diagnosis, or treatment.
        If you are struggling, please reach out to a qualified healthcare provider.
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("View My History ➡️", key="nav_pred_hist", use_container_width=True):
        st.session_state["page"] = "📜 My History"
        st.rerun()


# ═══════════════════════════════════════════════════════════
# PAGE: LOGIN / REGISTER
# ═══════════════════════════════════════════════════════════
elif st.session_state["page"] == "🔑 Login / Register":
    st.markdown("<br><br>", unsafe_allow_html=True)
    info_col, spacer, login_col = st.columns([1.1, 0.1, 1])
    
    with info_col:
        st.markdown("""
<div style="padding-right: 1.5rem;">
<div style='font-size:4.5rem; margin-bottom: 0.5rem; animation: float 3s ease-in-out infinite;'>🧠</div>
<h1 style="font-family: 'Poppins', sans-serif; font-size: 2.8rem; font-weight: 700; color: #1a1a2e; line-height: 1.15; margin-bottom: 1.2rem;">
Welcome to <br><span style="color: #6c63ff;">MindGuard.</span>
</h1>
<p style="color: #5a5a72; font-size: 1.05rem; line-height: 1.6; margin-bottom: 2.2rem; font-weight: 400;">
Your intelligent, AI-powered companion for mental health screening. 
Evaluate your psychological well-being across crucial life dimensions, securely track your history, and gain actionable insights.
</p>
<div style="display: flex; align-items: flex-start; margin-bottom: 1.5rem; background: #ffffff; padding: 1rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.03); border-left: 4px solid #6c63ff;">
<div style="font-size: 1.6rem; margin-right: 1.2rem; margin-top: 0.1rem;">🤖</div>
<div>
<h4 style="margin: 0 0 0.2rem 0; color: #1a1a2e; font-size: 1.05rem; font-weight: 600;">Data-Driven Insights</h4>
<p style="margin: 0; color: #7a7a8c; font-size: 0.9rem; line-height: 1.4;">Predict risk probabilities using advanced Machine Learning models trained on clinical datasets.</p>
</div>
</div>
<div style="display: flex; align-items: flex-start; margin-bottom: 1.5rem; background: #ffffff; padding: 1rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.03); border-left: 4px solid #43a047;">
<div style="font-size: 1.6rem; margin-right: 1.2rem; margin-top: 0.1rem;">📈</div>
<div>
<h4 style="margin: 0 0 0.2rem 0; color: #1a1a2e; font-size: 1.05rem; font-weight: 600;">Track Your Progress</h4>
<p style="margin: 0; color: #7a7a8c; font-size: 0.9rem; line-height: 1.4;">Save personalized assessments to monitor positive trends and download comprehensive PDF reports.</p>
</div>
</div>
<div style="display: flex; align-items: flex-start; background: #ffffff; padding: 1rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.03); border-left: 4px solid #e53935;">
<div style="font-size: 1.6rem; margin-right: 1.2rem; margin-top: 0.1rem;">🔐</div>
<div>
<h4 style="margin: 0 0 0.2rem 0; color: #1a1a2e; font-size: 1.05rem; font-weight: 600;">Private & Secure</h4>
<p style="margin: 0; color: #7a7a8c; font-size: 0.9rem; line-height: 1.4;">Your data is encrypted, strictly confidential, and managed through cutting-edge cloud infrastructure.</p>
</div>
</div>
</div>
""", unsafe_allow_html=True)
        
    with login_col:
        st.markdown("""
        <div style='text-align:center; margin-bottom: 0.2rem; margin-top: 0rem;'>
            <h2 style="font-family: 'Poppins', sans-serif; font-size: 1.9rem; color: #1a1a2e; font-weight: 600; margin-bottom: 0.1rem;">Get Started</h2>
            <p style="color: #7a7a8c; font-size: 0.9rem; margin-bottom: 0.2rem;">Sign in or create an account to continue.</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab_login, tab_register = st.tabs(["🔒 Existing User", "✨ New Account"])
        
        with tab_login:
            login_user = st.text_input("Username", key="login_user", placeholder="Enter your username")
            login_pass = st.text_input("Password", type="password", key="login_pass", placeholder="••••••••")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Sign In →", key="login_btn"):
                if not login_user or not login_pass:
                    st.error("Please fill all fields.")
                else:
                    success, uid, is_admin = db_utils.verify_login(login_user, login_pass)
                    if success:
                        st.toast("Welcome back!", icon="👋")
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = login_user
                        st.session_state["user_id"] = uid
                        st.session_state["is_admin"] = is_admin
                        st.session_state["page"] = "🛡️ Admin Dashboard" if is_admin else "🏠 Home"
                        st.rerun()
                    else:
                        st.error("Invalid credentials. Please try again.")

        with tab_register:
            reg_user = st.text_input("Choose Username", key="reg_user", placeholder="e.g. lilly_well")
            reg_pass = st.text_input("Create Password", type="password", key="reg_pass", placeholder="Min 6 characters")
            reg_pass2 = st.text_input("Confirm Password", type="password", key="reg_pass2", placeholder="Repeat password")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Account →", key="reg_btn"):
                if not reg_user or not reg_pass:
                    st.error("Please fill all fields.")
                elif reg_pass != reg_pass2:
                    st.error("Passwords do not match.")
                elif len(reg_pass) < 4: # Simple check
                    st.error("Password is too short.")
                else:
                    success, msg = db_utils.register_user(reg_user, reg_pass)
                    if success:
                        st.success("Account created! Switch to 'Existing User' to sign in.")
                    else:
                        st.error(f"Registration failed: {msg}")

        st.markdown("""
        <div style='text-align:center; margin-top: 1.5rem;'>
            <p style='font-size:0.75rem; color:#9a9aab;'>
                By accessing MindGuard, you agree to our terms of clinical screening ethics and privacy guidelines.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PAGE: ADMIN DASHBOARD
# ═══════════════════════════════════════════════════════════
elif st.session_state["page"] == "🛡️ Admin Dashboard" and st.session_state.get("is_admin"):
    st.markdown('<h1 class="page-title">Admin Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Platform-wide overview and assessment management.</p>', unsafe_allow_html=True)
    
    records = db_utils.get_all_assessments()
    
    # --- STATS CARDS ---
    c1, c2, c3 = st.columns(3)
    total_count = len(records)
    high_risk_count = sum(1 for r in records if r[3] == "High")
    unique_users = len(set(r[1] for r in records))
    
    with c1:
        st.markdown(f"""<div class="stat-card" style="border-left-color:#6c63ff;"><div class="stat-number">{total_count}</div><div class="stat-label">Total Assessments</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="stat-card" style="border-left-color:#e53935;"><div class="stat-number">{high_risk_count}</div><div class="stat-label">High Risk Cases</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="stat-card" style="border-left-color:#43a047;"><div class="stat-number">{unique_users}</div><div class="stat-label">Unique Evaluated Users</div></div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # --- SEARCH AND FILTER ---
    st.markdown('<p class="form-section-label">User Records</p>', unsafe_allow_html=True)
    search_q = st.text_input("🔍 Search by Username", placeholder="Enter username...").lower()
    
    if not records:
        st.info("No assessments recorded yet.")
    else:
        df_records = []
        for r in records:
            r_id, un, ts, risk, probs_str, inputs_str = r
            if search_q and search_q not in un.lower():
                continue
            dt = ts[:16].replace("T", " ")
            df_records.append({
                "ID": r_id,
                "Username": un,
                "DateTime": dt,
                "Risk Level": risk,
            })
        
        if not df_records:
            st.warning("No records match your search.")
        else:
            df = pd.DataFrame(df_records)
            
            # --- STYLED TABLE ---
            def color_risk(val):
                color = '#2e7d32' if val == 'Low' else ('#f57f17' if val == 'Moderate' else '#c62828')
                return f'color: {color}; font-weight: 600;'
            
            st.dataframe(df.style.applymap(color_risk, subset=['Risk Level']), use_container_width=True)
            
            # --- ACTIONS ---
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="form-section-label">Management Actions</p>', unsafe_allow_html=True)
            
            a1, a2 = st.columns([1, 1])
            with a1:
                target_id = st.number_input("Enter ID to Delete", min_value=0, step=1)
                if st.button("🗑️ Delete Record"):
                    if any(target_id == r['ID'] for r in df_records):
                        db_utils.delete_assessment(target_id)
                        st.success(f"Record {target_id} deleted successfully.")
                        st.rerun()
                    else:
                        st.error("Invalid ID.")
            
            with a2:
                st.markdown("<br>", unsafe_allow_html=True)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export to CSV",
                    data=csv,
                    file_name=f"mindguard_all_records_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )


# ═══════════════════════════════════════════════════════════
# PAGE: MY HISTORY
# ═══════════════════════════════════════════════════════════
elif st.session_state["page"] == "📜 My History" and st.session_state.get("logged_in"):
    import json as _json

    st.markdown('<h1 class="page-title">My Assessment History</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">A record of all your past mental health risk assessments.</p>', unsafe_allow_html=True)

    user_records = db_utils.get_user_assessments(st.session_state["user_id"])

    if not user_records:
        st.info("You haven't completed any assessments yet. Head to **🔮 Prediction** to get started!")
    else:
        # ── Summary stats ────────────────────────────────────
        total_u   = len(user_records)
        high_u    = sum(1 for r in user_records if r[3] == "High")
        mod_u     = sum(1 for r in user_records if r[3] == "Moderate")
        low_u     = sum(1 for r in user_records if r[3] == "Low")

        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.markdown(f'<div class="stat-card" style="border-left-color:#6c63ff;"><div class="stat-number">{total_u}</div><div class="stat-label">Total Assessments</div></div>', unsafe_allow_html=True)
        with sc2:
            st.markdown(f'<div class="stat-card" style="border-left-color:#43a047;"><div class="stat-number">{low_u}</div><div class="stat-label">Low Risk</div></div>', unsafe_allow_html=True)
        with sc3:
            st.markdown(f'<div class="stat-card" style="border-left-color:#f9a825;"><div class="stat-number">{mod_u}</div><div class="stat-label">Moderate Risk</div></div>', unsafe_allow_html=True)
        with sc4:
            st.markdown(f'<div class="stat-card" style="border-left-color:#e53935;"><div class="stat-number">{high_u}</div><div class="stat-label">High Risk</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Table ────────────────────────────────────────────
        st.markdown('<p class="form-section-label">All Records</p>', unsafe_allow_html=True)

        df_rows = []
        for r in user_records:
            r_id, un, ts, risk, probs_str, inputs_str = r
            dt = ts[:16].replace("T", " ") if ts else "—"
            try:
                probs = _json.loads(probs_str)
                high_pct = f"{probs[2]*100:.1f}%"
            except Exception:
                high_pct = "—"
            df_rows.append({
                "ID":         r_id,
                "Date / Time": dt,
                "Risk Level": risk,
                "High-Risk %": high_pct,
            })

        df_hist = pd.DataFrame(df_rows)

        def _color_risk(val):
            color = "#2e7d32" if val == "Low" else ("#f57f17" if val == "Moderate" else "#c62828")
            return f"color: {color}; font-weight: 600;"

        st.dataframe(
            df_hist.style.applymap(_color_risk, subset=["Risk Level"]),
            use_container_width=True,
        )

        # ── Trend chart ──────────────────────────────────────
        if len(user_records) >= 2:
            st.markdown("---")
            st.markdown("### 📈 Risk Trend Over Time")
            try:
                import plotly.graph_objects as go

                risk_num = {"Low": 0, "Moderate": 1, "High": 2}
                dates    = [r[2][:16].replace("T", " ") for r in reversed(user_records)]
                risks    = [risk_num.get(r[3], 0) for r in reversed(user_records)]
                colors_  = ["#43a047" if v == 0 else ("#f9a825" if v == 1 else "#e53935") for v in risks]

                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(
                    x=dates, y=risks,
                    mode="lines+markers",
                    line=dict(color="#6c63ff", width=2),
                    marker=dict(color=colors_, size=10, line=dict(color="#fff", width=2)),
                    hovertemplate="%{x}<br>Risk: %{customdata}<extra></extra>",
                    customdata=["Low" if v == 0 else ("Moderate" if v == 1 else "High") for v in risks],
                ))
                fig_trend.update_layout(
                    yaxis=dict(tickvals=[0, 1, 2], ticktext=["Low", "Moderate", "High"], range=[-0.3, 2.3]),
                    xaxis_title="Assessment Date",
                    height=300,
                    margin=dict(t=20, b=40, l=60, r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Poppins, sans-serif", size=11),
                )
                fig_trend.update_xaxes(showgrid=False)
                fig_trend.update_yaxes(gridcolor="#ede9e0")
                st.plotly_chart(fig_trend, use_container_width=True)
            except Exception:
                pass

        # ── Detail expander ──────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔍 Inspect a Past Assessment")
        selected_id = st.selectbox(
            "Select record by ID",
            options=[r[0] for r in user_records],
            format_func=lambda x: f"ID {x} — {next((r[2][:16].replace('T',' ') for r in user_records if r[0]==x), '')}",
        )
        chosen = next((r for r in user_records if r[0] == selected_id), None)
        if chosen:
            r_id, un, ts, risk, probs_str, inputs_str = chosen
            risk_css = {"Low": "result-low", "Moderate": "result-moderate", "High": "result-high"}.get(risk, "result-low")
            risk_emoji = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}.get(risk, "")
            st.markdown(f"""
            <div class="result-wrap {risk_css}" style="margin-top:0.8rem;">
                <div style="font-size:2rem;">{risk_emoji}</div>
                <div class="result-badge">{risk} Risk</div>
                <div class="result-desc">Assessed on {ts[:16].replace("T"," ") if ts else "—"}</div>
            </div>""", unsafe_allow_html=True)

            try:
                probs = _json.loads(probs_str)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<p class="form-section-label">📊 Probability Breakdown</p>', unsafe_allow_html=True)
                for lbl, fill_class, prob in zip(
                    ["Low", "Moderate", "High"],
                    ["prob-fill-low", "prob-fill-mod", "prob-fill-high"],
                    probs,
                ):
                    pct = prob * 100
                    st.markdown(f"""
                    <div class="prob-row">
                        <div class="prob-row-header"><span>{lbl}</span><span>{pct:.1f}%</span></div>
                        <div class="prob-track"><div class="{fill_class}" style="width:{pct}%"></div></div>
                    </div>""", unsafe_allow_html=True)
            except Exception:
                pass

            try:
                inputs = _json.loads(inputs_str)
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<p class="form-section-label">📋 Inputs Submitted</p>', unsafe_allow_html=True)
                inp_df = pd.DataFrame(list(inputs.items()), columns=["Feature", "Value"])
                st.dataframe(inp_df, use_container_width=True, hide_index=True)
            except Exception:
                pass

        # ── Export ───────────────────────────────────────────
        st.markdown("---")
        csv_hist = df_hist.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export My History to CSV",
            data=csv_hist,
            file_name=f"mindguard_history_{st.session_state['username']}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Learn more in About ➡️", key="nav_hist_about", use_container_width=True):
        st.session_state["page"] = "📖 About"
        st.rerun()


# ═══════════════════════════════════════════════════════════
# PAGE: ABOUT

# ═══════════════════════════════════════════════════════════
elif st.session_state["page"] == "📖 About":

    st.markdown('<h1 class="page-title">About MindGuard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Understanding mental health risk and how this tool works.</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🧠 Mental Health", "📐 How It Works", "⚠️ Risk Factors", "🌱 Prevention"])

    with tab1:
        st.markdown("""
        <div class="about-card">
            <h4>What is mental health risk?</h4>
            <p>Mental health risk refers to the likelihood that an individual may develop a mental health condition —
            such as anxiety, depression, or chronic stress — based on a combination of biological, psychological,
            and social factors. Early identification of risk can help people take preventive action before
            symptoms become severe.</p>
        </div>
        <div class="about-card">
            <h4>Risk categories</h4>
            <ul>
                <li><strong>Low Risk</strong> — Composite score ≤ 0.33. Indicators suggest stable mental well-being.</li>
                <li><strong>Moderate Risk</strong> — Score 0.33–0.66. Some stressors present; monitoring recommended.</li>
                <li><strong>High Risk</strong> — Score > 0.66. Multiple risk factors present; professional consultation advised.</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div class="about-card">
            <h4>Composite Score Construction</h4>
            <p>The target variable was derived from three clinical sub-scores: Anxiety (0–20), Depression (0–20),
            and Stress Level (0–9). Each was normalised to [0,1] and averaged into a single Composite Score,
            then thresholded into Low / Moderate / High.</p>
        </div>
        <div class="about-card">
            <h4>Machine Learning Pipeline</h4>
            <ul>
                <li><strong>Encoding:</strong> Ordinal mapping for ordered categoricals; numeric mapping for nominal ones.</li>
                <li><strong>Scaling:</strong> StandardScaler applied to continuous numeric features.</li>
                <li><strong>Class Balancing:</strong> SMOTENC handles mixed numeric/categorical training data.</li>
                <li><strong>Primary Model:</strong> XGBoost Classifier with <code>enable_categorical=True</code>,
                    tuned via 3-fold GridSearchCV optimising macro F1.</li>
                <li><strong>Comparison Models:</strong> Logistic Regression (SMOTENC), Random Forest (SMOTENC),
                    and MLP Classifier — all trained on the same balanced dataset.</li>
                <li><strong>Saved files:</strong> <code>mental_health_model_bundle.pkl</code> (XGBoost + scaler + columns),
                    <code>lr_model.pkl</code>, <code>rf_model.pkl</code>, <code>mlp_model.pkl</code>.</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with tab3:
        st.markdown("""
        <div class="about-card">
            <h4>Key risk factors</h4>
            <ul>
                <li><strong>Loneliness & low social support</strong> — among the strongest predictors of depression and anxiety.</li>
                <li><strong>Poor sleep</strong> — insufficient rest amplifies stress and emotional dysregulation.</li>
                <li><strong>Financial & work stress</strong> — chronic occupational stressors are leading contributors to burnout.</li>
                <li><strong>Low self-esteem</strong> — closely correlated with depressive episodes.</li>
                <li><strong>Family history</strong> — genetic predisposition increases vulnerability significantly.</li>
                <li><strong>Substance use</strong> — can both trigger and exacerbate mental health episodes.</li>
                <li><strong>Chronic illness</strong> — living with a long-term condition increases psychological burden.</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with tab4:
        st.markdown("""
        <div class="about-card">
            <h4>Evidence-based protective strategies</h4>
            <ul>
                <li>💤 <strong>Prioritise sleep</strong> — aim for 7–9 hours of consistent, quality rest nightly.</li>
                <li>🏃 <strong>Regular physical activity</strong> — 20–30 min/day significantly reduces anxiety and depression.</li>
                <li>🤝 <strong>Nurture social connections</strong> — meaningful relationships are one of the strongest buffers against poor mental health.</li>
                <li>🧘 <strong>Mindfulness & meditation</strong> — proven to reduce cortisol and improve emotional regulation.</li>
                <li>🗣️ <strong>Seek therapy</strong> — CBT and other evidence-based therapies are highly effective.</li>
                <li>🥗 <strong>Eat well</strong> — a diet rich in whole foods, omega-3s, and fibre supports brain health.</li>
                <li>🚫 <strong>Limit substance use</strong> — reducing alcohol and avoiding recreational drugs supports mental stability.</li>
                <li>📋 <strong>Regular screenings</strong> — self-monitoring and annual check-ins help catch changes early.</li>
            </ul>
        </div>
        <div class="disclaimer">
        If you or someone you know is in crisis, please contact a local mental health helpline or emergency services immediately.
        </div>""", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Home 🏠", key="nav_about_home", use_container_width=True):
            st.session_state["page"] = "🏠 Home"
            st.rerun()
    with col2:
        if st.button("Log Out 🚪", key="nav_about_logout", use_container_width=True):
            st.session_state["logged_in"] = False
            st.session_state["username"] = ""
            st.session_state["user_id"] = None
            st.session_state["is_admin"] = False
            st.session_state["page"] = "🔑 Login / Register"
            st.rerun()