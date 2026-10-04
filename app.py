import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="ChurnScope AI", page_icon="🔮", layout="centered")

st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #17171f 0%, #0a0a0f 100%);
    border: 1px solid rgba(245,158,11,.4);
    border-radius: 20px; padding: 36px 24px; text-align: center;
    margin-bottom: 24px; box-shadow: 0 0 50px rgba(245,158,11,.12);
}
.hero h1 {
    margin: 0; font-size: 2.4rem; font-weight: 800;
    background: linear-gradient(135deg, #fcd34d, #f59e0b);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { margin: 10px 0 0; color: #a1a1aa; font-size: 1rem; }
.card-danger {
    background: linear-gradient(135deg, #7f1d1d, #dc2626);
    color: #fff; border-radius: 18px; padding: 30px; text-align: center;
    box-shadow: 0 10px 30px rgba(220,38,38,.35); margin: 18px 0;
    border: 1px solid rgba(255,255,255,.15);
}
.card-safe {
    background: linear-gradient(135deg, #065f46, #10b981);
    color: #fff; border-radius: 18px; padding: 30px; text-align: center;
    box-shadow: 0 10px 30px rgba(16,185,129,.35); margin: 18px 0;
    border: 1px solid rgba(255,255,255,.15);
}
.card-danger h2, .card-safe h2 { margin: 0 0 6px; font-size: 1.7rem; }
.badge {
    display: inline-block; background: rgba(0,0,0,.3); border-radius: 999px;
    padding: 6px 18px; font-weight: 700; font-size: .85rem; margin-bottom: 10px;
    border: 1px solid rgba(255,255,255,.25);
}
.stButton > button {
    background: linear-gradient(135deg, #fbbf24, #f59e0b) !important;
    color: #000 !important; border: none !important; border-radius: 12px !important;
    padding: .7rem 2rem !important; font-weight: 800 !important; width: 100%;
    box-shadow: 0 4px 20px rgba(245,158,11,.35) !important;
}
div[data-testid="stMetricValue"] { color: #fbbf24 !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open("churn_model.pkl", "rb") as f:
        d = pickle.load(f)
    return d["model"], d["encoders"], d["columns"]

model, encoders, columns = load_model()

st.markdown('<div class="hero"><h1>🔮 ChurnScope AI</h1><p>Predict which customers will leave — before they do.</p></div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.metric("🤖 Model", "Random Forest")
c2.metric("🎯 Accuracy", "71%")
c3.metric("📊 Trained on", "5,000 customers")

st.markdown("### 📝 Customer Details")

with st.form("f"):
    a, b = st.columns(2)
    with a:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 1, 72, 12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
    with b:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        monthly = st.number_input("Monthly Charges ($)", 10.0, 200.0, 70.0)
        total = st.number_input("Total Charges ($)", 10.0, 15000.0, 1000.0)
    go = st.form_submit_button("🔮 Predict Churn")

if go:
    row = {
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone, "InternetService": internet, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment,
        "MonthlyCharges": monthly, "TotalCharges": total,
    }
    df = pd.DataFrame([row])
    for col, le in encoders.items():
        df[col] = le.transform(df[col])
    df = df[columns]
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1] * 100

    if pred == 1:
        st.markdown(f'<div class="card-danger"><div class="badge">⚠️ HIGH RISK</div><h2>Customer Will CHURN</h2><p style="font-size:1.2rem;">{prob:.1f}% probability of leaving</p></div>', unsafe_allow_html=True)
        st.warning("💡 **Retention tip:** Offer a discount, free upgrade, or move them to a yearly contract!")
    else:
        st.markdown(f'<div class="card-safe"><div class="badge">✅ LOW RISK</div><h2>Customer Will STAY</h2><p style="font-size:1.2rem;">{100-prob:.1f}% likely to stay</p></div>', unsafe_allow_html=True)
        st.info("💡 **Tip:** Keep them happy — maybe a loyalty reward!")

    st.markdown("#### Churn Probability")
    st.progress(int(prob))

st.markdown("---")
st.caption("Built with Random Forest • scikit-learn • Streamlit | by Engineer Muhammad Sheharyar Khan")
