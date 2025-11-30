# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.title("⚙️ Predictive Maintenance - RUL Prediction")
st.markdown("Simulates machine sensor data, trains a Random Forest, and predicts Remaining Useful Life (RUL).")

# -----------------------------
# Simulate Data
# -----------------------------
@st.cache_data
def simulate_data():
    np.random.seed(42)
    machine_ids = np.repeat(np.arange(1,6), 200)
    time_steps = np.tile(np.arange(200), 5)
    temperature = np.random.normal(70, 5, 1000) + time_steps*0.05
    vibration = np.random.normal(0.5, 0.1, 1000) + time_steps*0.002
    RUL = 200 - time_steps + np.random.normal(0,5,1000)
    
    df = pd.DataFrame({
        "MachineID": machine_ids,
        "Time": time_steps,
        "Temperature": temperature,
        "Vibration": vibration,
        "RUL": RUL
    })
    
    # Feature engineering: rolling stats
    df['Temp_RollingMean'] = df.groupby('MachineID')['Temperature'].transform(lambda x: x.rolling(5,1).mean())
    df['Temp_RollingStd'] = df.groupby('MachineID')['Temperature'].transform(lambda x: x.rolling(5,1).std())
    df['Vib_RollingMean'] = df.groupby('MachineID')['Vibration'].transform(lambda x: x.rolling(5,1).mean())
    df['Vib_RollingStd'] = df.groupby('MachineID')['Vibration'].transform(lambda x: x.rolling(5,1).std())
    
    df = df.dropna()
    return df

df = simulate_data()

# -----------------------------
# Sidebar: Select Machine
# -----------------------------
st.sidebar.subheader("Controls")
machine = st.sidebar.selectbox("Select Machine ID", df["MachineID"].unique())

df_machine = df[df["MachineID"]==machine]

# -----------------------------
# Train-Test Split
# -----------------------------
X = df_machine[['Temperature','Vibration','Temp_RollingMean','Temp_RollingStd','Vib_RollingMean','Vib_RollingStd']]
y = df_machine['RUL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
st.markdown(f"**RMSE:** {rmse:.2f}")
st.markdown(f"**R² Score:** {r2:.2f}")

# -----------------------------
# Visualize Predictions
# -----------------------------
st.subheader(f"Machine {machine} - Actual vs Predicted RUL")
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(df_machine["Time"], df_machine["RUL"], label="Actual RUL", color="blue", linewidth=2)
ax.plot(df_machine["Time"], model.predict(X), label="Predicted RUL", color="red", linestyle="--")
ax.set_xlabel("Time")
ax.set_ylabel("Remaining Useful Life")
ax.legend()
st.pyplot(fig)

# -----------------------------
# Feature Importance
# -----------------------------
st.subheader("Feature Importance")
importance = model.feature_importances_
features = X.columns
feature_df = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=False)
st.bar_chart(feature_df.set_index('Feature'))

# -----------------------------
# Recent Data Table
# -----------------------------
st.subheader("Recent Sensor Data and Predicted RUL")
df_machine['RUL_Pred'] = model.predict(X)
st.dataframe(df_machine.tail(10))

# -----------------------------
# Download CSV
# -----------------------------
csv = df_machine.to_csv(index=False)
st.download_button(
    label="📥 Download Machine Data & Predictions CSV",
    data=csv,
    file_name=f'machine_{machine}_RUL.csv',
    mime='text/csv'
)
