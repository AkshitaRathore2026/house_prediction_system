import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os



# Load Model
model = joblib.load("models/house_price_model.pkl")
model_scores = joblib.load("models/model_scores.pkl")
best_model_name = joblib.load("models/best_model_name.pkl")

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)





# =========================
# CUSTOM CSS STYLING
# =========================

st.markdown("""
<style>
.big-font {
        font-size:32px !important;
        font-weight:700;
        color:#00C853;
        text-align:center;
}

.stButton>button {
            background-color:#00C853;
            color:white;
            border-radius:10px;
            height:50px;
            width:100%;
            font-size:18px;
        }
 
            
.stDownloadButton>button {
                    background-color:#2962FF;
                    color:white;
                    border-radius:10px;
                }
</style>
""", unsafe_allow_html=True)






st.title("🏠 House Price Prediction")


st.markdown(
    '<p class="big-font">AI Powered Real Estate Prediction System</p>',
    unsafe_allow_html=True
)





st.subheader("📊 Model Performance")
col1, col2, col3, col4 = st.columns(4)
scores = list(model_scores.items())
with col1:
    st.metric(
            scores[0][0],
            f"{scores[0][1] * 100:.2f}%"
        )

with col2:
    st.metric(
            scores[1][0],
            f"{scores[1][1] * 100:.2f}%"
        )

with col3:
    st.metric(
            scores[2][0],
            f"{scores[2][1] * 100:.2f}%"
        )
    
with col4:
    st.metric(
            scores[3][0],
            f"{scores[3][1] * 100:.2f}%"
    )



st.success(
            f"✅ Best Model Used: {best_model_name}"
        )





chart_data = pd.DataFrame({
    'Model': list(model_scores.keys()),
    'Accuracy': list(model_scores.values())
})

st.bar_chart(
    chart_data.set_index('Model')
)





st.write("Enter House Details")


# User Inputs
area = st.number_input("Area (sq ft)", min_value=500, step=50)

bedrooms = st.number_input("Bedrooms", min_value=1)

bathrooms = st.number_input("Bathrooms", min_value=1)

stories = st.number_input("Stories", min_value=1)

mainroad = st.selectbox("Main Road", ["yes", "no"])
#mainroad = 1 if mainroad == "yes" else 0

guestroom = st.selectbox("Guest Room", ["yes", "no"])
#guestroom = 1 if guestroom == "yes" else 0

basement = st.selectbox("Basement", ["yes", "no"])
#basement = 1 if basement == "yes" else 0

hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
#hotwaterheating = 1 if hotwaterheating == "yes" else 0

airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
#airconditioning = 1 if airconditioning == "yes" else 0

parking = st.number_input("Parking", min_value=0)

prefarea = st.selectbox("Preferred Area", ["yes", "no"])
#prefarea = 1 if prefarea == "yes" else 0

furnishingstatus = st.selectbox(
                            "Furnishing Status",
                            ["furnished", "semi-furnished", "unfurnished"]
                        )






# =========================
# FEATURE ENGINEERING
# =========================

area_per_bedroom = area / (bedrooms + 1)

total_rooms = bedrooms + bathrooms

luxury_score = (
            (1 if airconditioning == "yes" else 0) +
            (1 if hotwaterheating == "yes" else 0) +
            (1 if guestroom == "yes" else 0)
        )

parking_per_bedroom = (parking / (bedrooms + 1))

area_per_story = (area / (stories + 1))






# =========================
# CREATE INPUT DATAFRAME
# =========================

input_data = pd.DataFrame({
    'area': [area],
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'stories': [stories],
    'mainroad': [mainroad],
    'guestroom': [guestroom],
    'basement': [basement],
    'hotwaterheating': [hotwaterheating],
    'airconditioning': [airconditioning],
    'parking': [parking],
    'prefarea': [prefarea],
    'furnishingstatus': [furnishingstatus],
    'area_per_bedroom': [area_per_bedroom],
    'total_rooms': [total_rooms],
    'luxury_score': [luxury_score],
    'parking_per_bedroom': [parking_per_bedroom],
    'area_per_story': [area_per_story]
})






# =========================
# PREDICTION
# =========================

if st.button("Predict Price"):
    prediction = np.expm1(model.predict(input_data))

    os.makedirs("reports", exist_ok=True)

    history = pd.DataFrame({
                    "Area": [area],
                    "Bedrooms": [bedrooms],
                    "Bathrooms": [bathrooms],
                    "Predicted Price": [prediction[0]]
                })
    
    history_file = "reports/prediction_history.csv"
    history.to_csv(
                history_file,
                mode='a',
                header=not os.path.exists(history_file),
                index=False
            )

    st.success(f"Predicted Price: ₹ {prediction[0]:,.2f}")

    report = input_data.copy()
    report["Predicted Price"] = prediction[0]
    csv = report.to_csv(index=False)

    st.download_button(
                label="Download Prediction Report",
                data=csv,
                file_name="prediction_report.csv",
                mime="text/csv"
            )





-
# =========================
# SHOW PREDICTION HISTORY
# =========================

if os.path.exists("reports/prediction_history.csv"):

    history_df = pd.read_csv("reports/prediction_history.csv")

    st.subheader("📜 Prediction History")

    st.dataframe(history_df)
