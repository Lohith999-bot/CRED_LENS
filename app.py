from flask import Flask, render_template, request, send_file
import numpy as np
import pickle
import os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)

# Load trained model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

model = pickle.load(open(model_path, "rb"))
# -------------------------------
# Recommendation Logic
# -------------------------------
def recommend_loan(income, existing_loans, tenure):
    existing_emi = existing_loans * 5000
    max_emi = income * 0.3
    available_emi = max_emi - existing_emi

    if available_emi <= 0:
        return 0

    return int(available_emi * tenure)

# -------------------------------
# Reason + Suggestion Logic
# -------------------------------
def analyze_loan(credit_score, emi_ratio, debt_ratio, previous_default):
    reasons = []
    suggestions = []

    if credit_score < 600:
        reasons.append("Low credit score")
        suggestions.append("Improve credit score above 650")

    if emi_ratio > 0.4:
        reasons.append("High EMI to income ratio")
        suggestions.append("Reduce loan amount or increase tenure")

    if debt_ratio > 0.6:
        reasons.append("High debt to income ratio")
        suggestions.append("Clear existing debts")

    if previous_default == 1:
        reasons.append("Previous loan default history")
        suggestions.append("Maintain clean repayment record")

    if len(reasons) == 0:
        reasons.append("Strong financial profile")
        suggestions.append("Eligible for higher loan amount")

    return reasons, suggestions

# -------------------------------
# PDF REPORT
# -------------------------------
def generate_report(data, status, recommended, reasons, suggestions):
    file_path = "loan_report.pdf"

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("Loan Assessment Report", styles['Title']))
    content.append(Spacer(1, 20))

    for key, value in data.items():
        content.append(Paragraph(f"{key}: {value}", styles['Normal']))

    content.append(Spacer(1, 15))
    content.append(Paragraph(f"Status: {status}", styles['Normal']))
    content.append(Paragraph(f"Recommended Loan: ₹ {recommended}", styles['Normal']))

    content.append(Spacer(1, 15))
    content.append(Paragraph("Reasons:", styles['Heading2']))
    for r in reasons:
        content.append(Paragraph(f"- {r}", styles['Normal']))

    content.append(Spacer(1, 10))
    content.append(Paragraph("Suggestions:", styles['Heading2']))
    for s in suggestions:
        content.append(Paragraph(f"- {s}", styles['Normal']))

    doc.build(content)

    return file_path

# -------------------------------
# SAFE INPUT HANDLING
# -------------------------------
def get_float(form, key):
    try:
        value = form.get(key, "").strip()
        return float(value) if value != "" else 0.0
    except:
        return 0.0

def get_int(form, key):
    try:
        value = form.get(key, "").strip()
        return int(value) if value != "" else 0
    except:
        return 0

# -------------------------------
# ROUTES
# -------------------------------

@app.route("/")
def home():
    return render_template("page0.html")

@app.route("/form")
def form():
    return render_template("page1.html") 

@app.route("/predict", methods=["POST"])
def predict():

    # ✅ Safe inputs
    income = get_float(request.form, "income")
    age = get_float(request.form, "age")
    loan_required = get_float(request.form, "loan_required")
    existing_total_loan = get_float(request.form, "existing_total_loan")
    tenure = get_float(request.form, "tenure")
    credit_score = get_float(request.form, "credit_score")
    previous_default = get_int(request.form, "previous_default")
    existing_loans = get_int(request.form, "existing_loans")

    # ✅ Validation
    if income == 0 or tenure == 0:
        return "Error: Income and tenure must be greater than 0"

    # 🔥 Calculate ratios (FIXED INDENTATION)
    loan_to_income_ratio = loan_required / income
    emi = loan_required / tenure
    emi_to_income_ratio = emi / income
    existing_emi = existing_total_loan / tenure
    debt_to_income_ratio = (existing_emi + emi) / income

    # Model input
    features = np.array([[income, age, loan_required,
                          existing_total_loan, tenure,
                          loan_to_income_ratio,
                          emi_to_income_ratio,
                          debt_to_income_ratio,
                          credit_score,
                          previous_default,
                          existing_loans]])

    prediction = model.predict(features)

    if prediction[0] == 0:
        status = "Approved"
        recommended = recommend_loan(income, existing_loans, tenure)
    else:
        status = "Rejected"
        recommended = 0

    # Analyze reasons
    reasons, suggestions = analyze_loan(
        credit_score,
        emi_to_income_ratio,
        debt_to_income_ratio,
        previous_default
    )

    # Prepare data for report
    data = {
        "Income": income,
        "Age": age,
        "Loan Required": loan_required,
        "Existing Loan": existing_total_loan,
        "Tenure": tenure,
        "Credit Score": credit_score
    }

    # Generate PDF
    generate_report(data, status, recommended, reasons, suggestions)

    return render_template("result.html",
                           status=status,
                           recommended=recommended,
                           lti=round(loan_to_income_ratio, 2),
                           eti=round(emi_to_income_ratio, 2),
                           dti=round(debt_to_income_ratio, 2),
                           reasons=reasons,
                           suggestions=suggestions)

@app.route("/download")
def download():
    return send_file("loan_report.pdf", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)