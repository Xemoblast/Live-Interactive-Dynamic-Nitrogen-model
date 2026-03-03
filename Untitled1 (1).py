
import os
print(os.getcwd())

import streamlit as st
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="Nitrogen Dynamic Model", layout="wide")

def nitrogen_model(y, t, params):
    U, NH3, NH4, NO2, NO3, N_plant, L_vol, L_leach, L_den = y
    k_u_ref, k_eq, k_vol_ref, k_nit1_ref, k_nit2_ref, k_upt_NH4, k_upt_NO3, k_leach_ref, k_den_ref, theta = params
    
    f_w_aerobic = np.exp(-0.5 * ((theta - 0.30) / 0.15)**2)
    f_w_denit = 1 / (1 + np.exp(-20 * (theta - 0.38)))
    f_w_leach = max(0, (theta - 0.30) / 0.15)

    k_u = k_u_ref * f_w_aerobic
    k_vol = k_vol_ref * (1.5 - f_w_aerobic)
    k_nit1 = k_nit1_ref * f_w_aerobic
    k_nit2 = k_nit2_ref * f_w_aerobic
    k_leach = k_leach_ref * f_w_leach
    k_den = k_den_ref * f_w_denit

    dU_dt = -k_u * U
    dNH3_dt = k_u * U - k_eq * NH3 - k_vol * NH3
    dNH4_dt = k_eq * NH3 - k_nit1 * NH4 - k_upt_NH4 * NH4
    dNO2_dt = k_nit1 * NH4 - k_nit2 * NO2
    dNO3_dt = k_nit2 * NO2 - k_upt_NO3 * NO3 - k_leach * NO3 - k_den * NO3
    dNplant_dt = k_upt_NH4 * NH4 + k_upt_NO3 * NO3
    dLvol_dt = k_vol * NH3
    dLleach_dt = k_leach * NO3
    dLden_dt = k_den * NO3

    return [dU_dt, dNH3_dt, dNH4_dt, dNO2_dt, dNO3_dt, dNplant_dt, dLvol_dt, dLleach_dt, dLden_dt]

if 'params' not in st.session_state:
    st.session_state.urea = 150.0
    st.session_state.theta = 0.30
    st.session_state.ku = 0.07

st.title("Live Interactive Nitrogen Dynamics Model")
st.markdown("Use the presets below to instantly simulate environmental conditions, or use the sidebar for fine-tuned control.")

col1, col2, col3, col4, col5 = st.columns(5)

if col1.button("Standard Baseline"):
    st.session_state.urea, st.session_state.theta, st.session_state.ku = 150.0, 0.30, 0.07
if col2.button("Extreme Drought"):
    st.session_state.urea, st.session_state.theta, st.session_state.ku = 150.0, 0.08, 0.09
if col3.button("Severe Flooding"):
    st.session_state.urea, st.session_state.theta, st.session_state.ku = 150.0, 0.50, 0.05
if col4.button("Fertilizer Overload"):
    st.session_state.urea, st.session_state.theta, st.session_state.ku = 450.0, 0.32, 0.07
if col5.button("Fast Kinetics"):
    st.session_state.urea, st.session_state.theta, st.session_state.ku = 150.0, 0.30, 0.20

st.sidebar.header("Manual Adjustments")
urea_input = st.sidebar.slider("Urea Applied (kg/ha)", 0.0, 600.0, st.session_state.urea)
theta = st.sidebar.slider("Soil Moisture (θ)", 0.0, 1.0, st.session_state.theta)
k_u_ref = st.sidebar.slider("Hydrolysis Rate (k_u)", 0.0, 0.5, st.session_state.ku)
days_to_plot = st.sidebar.slider("Simulation Length (Days)", 1, 60, 30)

U0 = urea_input * 0.46
y_init = [U0, 0, 0, 0, 0, 0, 0, 0, 0]
# Fixed reference params based on your research scenarios
params = [k_u_ref, 0.5, 0.005, 0.01, 0.02, 0.008, 0.012, 0.02, 0.01, theta]

t = np.linspace(0, days_to_plot * 24, 1000)
sol = odeint(nitrogen_model, y_init, t, args=(params,))

fig, ax = plt.subplots(figsize=(10, 5))
U_p, NH3_p, NH4_p, NO2_p, NO3_p, Np_p, Lv_p, Ll_p, Ld_p = sol.T

ax.plot(t/24, U_p, label="Urea (N)", color='teal', lw=2)
ax.plot(t/24, NH4_p, label="Ammonium", color='royalblue')
ax.plot(t/24, NO3_p, label="Nitrate", color='crimson')
ax.plot(t/24, Np_p, label="Plant N", color='green', lw=3)
ax.fill_between(t/24, Lv_p + Ll_p + Ld_p, color='grey', alpha=0.1, label="Total Losses")

ax.set_xlabel("Time (Days)")
ax.set_ylabel("Nitrogen (kg N / ha)")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.2)
st.pyplot(fig)

st.subheader("Analysis & Efficiency")
final = sol[-1]
total_loss = final[6] + final[7] + final[8]
efficiency = (final[5] / U0) * 100 if U0 > 0 else 0

m1, m2, m3 = st.columns(3)
m1.metric("Plant Uptake", f"{final[5]:.2f} kg N/ha")
m2.metric("Environmental Loss", f"{total_loss:.2f} kg N/ha", delta_color="inverse")
m3.metric("N-Use Efficiency (NUE)", f"{efficiency:.1f}%")

st.write("---")
st.caption("Model based on first-order kinetics and moisture-scaling functions validated against Pacholski et al. (2019).")
