# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -----------------------------------------------------
# 1. Load and Preprocess Data
# -----------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    # Drop customerID
    data.drop('customerID', axis=1, inplace=True)
    
    # Convert TotalCharges to numeric
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
    
    # Encode categorical columns
    for col in data.select_dtypes(include=['object']).columns:
        if col != 'Churn':
            data[col] = LabelEncoder().fit_transform(data[col])
    
    # Encode target
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    
    return data

data = load_data()

# Features & Target
X = data.drop('Churn', axis=1)
y = data['Churn']

# Split & Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -----------------------------------------------------
# 2. Streamlit Dashboard
# -----------------------------------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction Analysis")

# app.py

# Sidebar Navigation

if "page" not in st.session_state:
    st.session_state.page = "Home"

page = st.sidebar.radio("Go to:", ["Home", "EDA", "Feature Importance", "Predict Churn"], 
                        index=["Home", "EDA", "Feature Importance", "Predict Churn"].index(st.session_state.page))


# -----------------------------------------------------
# 🏠 Home Page
# -----------------------------------------------------
if page == "Home":
    st.markdown("""
    Welcome to the **Customer Churn Prediction App** 📊     
    This dashboard helps businesses:
    - Understand customer churn patterns 🔍  
    - Identify the top factors that drive churn 🚨  
    - Predict whether a customer will leave or stay 💡  

    **Why it matters?**  
    Retaining customers is often cheaper than acquiring new ones.  
    Predicting churn helps reduce revenue loss and improve customer satisfaction.
    """)

    # Show Key Metrics
    col1, col2, col3 = st.columns(3)

    churn_rate = (data['Churn'].mean()) * 100
    avg_tenure = data['tenure'].mean()
    avg_monthly = data['MonthlyCharges'].mean()

    col1.metric("Churn Rate", f"{churn_rate:.2f}%")
    col2.metric("Avg. Tenure", f"{avg_tenure:.1f} months")
    col3.metric("Avg. Monthly Charges", f"${avg_monthly:.2f}")

    # Image
    st.image("churn.jpg", caption="Customer Churn Analysis")

    # 🔗 Button to navigate to Predict Churn page
    if st.button("🔮 Go to Prediction Page"):
        st.session_state.page = "Predict Churn"



# -----------------------------------------------------
# 3. Exploratory Data Analysis
# -----------------------------------------------------
if page == "EDA":
    st.subheader("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Churn Distribution")
        churn_counts = data['Churn'].value_counts()
        fig, ax = plt.subplots()
        sns.barplot(x=churn_counts.index, y=churn_counts.values, palette="coolwarm", ax=ax)
        ax.set_xticklabels(["No Churn", "Churn"])
        st.pyplot(fig)

    with col2:
        st.write("### Monthly Charges vs Churn")
        fig, ax = plt.subplots()
        sns.boxplot(x="Churn", y="MonthlyCharges", data=data, palette="coolwarm", ax=ax)
        ax.set_xticklabels(["No Churn", "Churn"])
        st.pyplot(fig)

# -----------------------------------------------------
# 4. Feature Importance
# -----------------------------------------------------
elif page == "Feature Importance":
    st.subheader("Feature Importance from Random Forest")
    importances = model.feature_importances_
    features = X.columns
    fi = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=fi, palette="viridis", ax=ax)
    st.pyplot(fig)

# -----------------------------------------------------
# 5. Prediction Page
# -----------------------------------------------------
# -----------------------------------------------------
# 5. Prediction Page
# -----------------------------------------------------
else:
    st.subheader("🔮 Predict Customer Churn")

    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

    with col2:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    
    # Map inputs (same encoding as training)
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
    payment_map = {
        "Electronic check": 0,
        "Mailed check": 1,
        "Bank transfer (automatic)": 2,
        "Credit card (automatic)": 3
    }

    features = np.array([[tenure, monthly_charges, total_charges,
                          contract_map[contract], internet_map[internet_service], payment_map[payment_method]]])

    # Pad with zeros for missing features
    features = np.pad(features, ((0,0),(0,X.shape[1]-features.shape[1])), 'constant')

    if st.button("Predict Churn"):
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1]

        if pred == 1:
            st.error(f"❌ This customer is likely to **churn**. (Probability: {prob:.2f})")
            
            # 🔹 Show Recommendations
            st.markdown("### 💡 Recommendations to Reduce Churn")
            st.write("- Offer discounts or loyalty rewards to retain this customer 🎁")
            st.write("- Provide personalized customer support and resolve issues quickly 📞")
            st.write("- Suggest moving to a longer contract for better benefits 📜")
            st.write("- Improve internet service quality if using Fiber optic ⚡")

        else:
            st.success(f"✅ This customer is likely to **stay**. (Probability: {prob:.2f})")
            
            # 🔹 Show Engagement Tips
            st.markdown("### 💡 Recommendations to Maintain Loyalty")
            st.write("- Continue providing quality service to maintain satisfaction ✅")
            st.write("- Offer upselling opportunities (e.g., premium plans) 📈")
            st.write("- Send personalized thank-you emails or rewards 💌")
