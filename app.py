import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- إعدادات الصفحة ---
st.set_page_config(
    page_title="Gym Churn Intelligence AI | Method X & Form50",
    page_icon="🏋️‍♂️",
    layout="wide"
)

# --- تصميم CSS عالمي وفاخر ---
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8F4B 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        font-weight: bold;
        border-radius: 8px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-2px);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        text-align: center;
        border-left: 5px solid #FF4B4B;
    }
    h1, h2, h3 {
        color: #1E1E1E;
        font-family: 'Helvetica Neue', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# --- عنوان التطبيق ---
st.markdown("<h1 style='text-align: center;'>🏋️‍♂️ AI-Powered Gym Member Churn Analytics</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Predict member drop-offs, retain high-value clients, and export automated retention reports.</p>", unsafe_allow_html=True)
st.markdown("---")

# --- دالة إنشاء داتا سيت افتراضية ذكية للجيمات ---
@st.cache_data
def load_default_data():
    np.random.seed(42)
    n_samples = 300
    data = {
        'MemberID': [f'MEM-{1000+i}' for i in range(n_samples)],
        'Age': np.random.randint(18, 65, size=n_samples),
        'MembershipType': np.random.choice(['Monthly', 'Annual', 'VIP'], size=n_samples, p=[0.5, 0.3, 0.2]),
        'AttendancePerWeek': np.random.randint(0, 7, size=n_samples),
        'AvgWorkoutDurationMinutes': np.random.randint(30, 120, size=n_samples),
        'DaysSinceLastVisit': np.random.randint(0, 30, size=n_samples),
        'FeedbackScore': np.random.randint(1, 6, size=n_samples)
    }
    df = pd.DataFrame(data)
    # محاكاة منطقية لاحتمالية الانسحاب (Churn)
    churn_condition = (df['AttendancePerWeek'] < 2) | (df['DaysSinceLastVisit'] > 14) | (df['FeedbackScore'] < 3)
    df['Churn'] = np.where(churn_condition & (np.random.rand(n_samples) > 0.3), 1, 0)
    return df

# --- الشريط الجانبي لرفع البيانات أو استخدام الافتراضية ---
st.sidebar.header("📁 Data Source Management")
uploaded_file = st.sidebar.file_uploader("Upload Gym CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom dataset loaded successfully!")
else:
    df = load_default_data()
    st.sidebar.info("Using built-in simulation dataset (ideal for demos).")

# --- معالجة البيانات وتدريب نموذج سريع ---
features = ['Age', 'AttendancePerWeek', 'AvgWorkoutDurationMinutes', 'DaysSinceLastVisit', 'FeedbackScore']

# تحويل البيانات النصية إذا وجدت
df_model = pd.get_dummies(df[features], drop_first=True)
X = df_model
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# حساب التنبؤات لكل الأعضاء
df['Churn_Probability'] = model.predict_proba(X)[:, 1]
df['Risk_Level'] = pd.cut(df['Churn_Probability'], bins=[-0.1, 0.4, 0.7, 1.1], labels=['Low Risk 🟢', 'Medium Risk 🟡', 'High Risk 🔴'])

# --- عرض مؤشرات الأداء الرئيسية (KPIs) ---
total_members = len(df)
high_risk_count = (df['Risk_Level'] == 'High Risk 🔴').sum()
churn_rate = (y.sum() / total_members) * 100

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""<div class="metric-card"><h3>Total Members</h3><h2>{total_members}</h2></div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card"><h3>At-Risk Members</h3><h2 style='color: #FF4B4B;'>{high_risk_count}</h2></div>""", unsafe_allow_html=True)
with col3:
    st.markdown(f"""<div class="metric-card"><h3>Overall Churn Rate</h3><h2>{churn_rate:.1f}%</h2></div>""", unsafe_allow_html=True)

st.markdown("---")

# --- جدول الأعضاء مع مستويات الخطر ---
st.subheader("🔍 Member Churn Risk Intelligence Table")
st.dataframe(df[['MemberID', 'Age', 'MembershipType', 'AttendancePerWeek', 'DaysSinceLastVisit', 'Churn_Probability', 'Risk_Level']], use_container_width=True)

# --- زر تصدير التقرير المفصل ---
st.markdown("### 📄 Export Retention Action Report")
st.markdown("Download a clean analytical CSV report containing high-risk members and automated retention insights to present directly to gym management.")

# تجهيز ملف التقرير للتصدير
report_df = df[['MemberID', 'MembershipType', 'AttendancePerWeek', 'DaysSinceLastVisit', 'Churn_Probability', 'Risk_Level']]
csv_report = report_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download Detailed Churn Action Report (CSV)",
    data=csv_report,
    file_name="Gym_Retention_Action_Report.csv",
    mime="text/csv"
)

# --- قسم التنبؤ الفردي لعضو جديد ---
st.markdown("---")
st.subheader("⚡ Real-time Individual Member Churn Predictor")
with st.form("prediction_form"):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        input_age = st.number_input("Member Age", 18, 80, 28)
        input_attendance = st.number_input("Attendance Per Week", 0, 7, 2)
    with col_b:
        input_duration = st.number_input("Workout Duration (Mins)", 20, 180, 60)
        input_days = st.number_input("Days Since Last Visit", 0, 30, 5)
    with col_c:
        input_feedback = st.slider("Feedback Score (1-5)", 1, 5, 3)
    
    submit_btn = st.form_submit_button("Analyze Member Risk")

if submit_btn:
    input_data = pd.DataFrame([[input_age, input_attendance, input_duration, input_days, input_feedback]], columns=features)
    # مطابقة الأعمدة مع الموديل
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)
    
    prob = model.predict_proba(input_encoded)[0][1]
    
    if prob > 0.7:
        st.error(f"🚨 High Churn Risk Detected! Probability: {prob*100:.1f}%. Immediate intervention required (Offer loyalty discount or personal trainer session).")
    elif prob > 0.4:
        st.warning(f"⚠️ Medium Risk. Probability: {prob*100:.1f}%. Monitor attendance closely.")
    else:
        st.success(f"✅ Safe / Active Member. Probability of leaving: {prob*100:.1f}%.")