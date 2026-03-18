import streamlit as st
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

tab1, tab2 = st.tabs(["Interactive Simulator", "Inverse Prediction Mode"])

with tab1:
    if 'urea' not in st.session_state:
        st.session_state.urea = 150.0
        st.session_state.theta = 0.30
        st.session_state.ku = 0.07

    st.title("Live Interactive Nitrogen Dynamics Model")
    
    c1, c2, c3, c4, c5 = st.columns(5)
    if c1.button("Standard Baseline"):
        st.session_state.urea, st.session_state.theta, st.session_state.ku = 150.0, 0.30, 0.07
    if c2.button("Extreme Drought"):
        st.session_state.urea, st.session_state.theta, st.session_state.ku = 150.0, 0.08, 0.09
    if c3.button("Severe Flooding"):
        st.session_state.urea, st.session_state.theta, st.session_state.ku = 150.0, 0.50, 0.05
    if c4.button("Fertilizer Overload"):
        st.session_state.urea, st.session_state.theta, st.session_state.ku = 450.0, 0.32, 0.07
    if c5.button("Fast Kinetics"):
        st.session_state.urea, st.session_state.theta, st.session_state.ku = 150.0, 0.30, 0.20

    st.sidebar.header("Manual Adjustments")
    u_in = st.sidebar.slider("Urea Applied (kg/ha)", 0.0, 600.0, st.session_state.urea)
    th_in = st.sidebar.slider("Soil Moisture (θ)", 0.0, 1.0, st.session_state.theta)
    ku_in = st.sidebar.slider("Hydrolysis Rate (k_u)", 0.0, 0.5, st.session_state.ku)
    days = st.sidebar.slider("Simulation Length (Days)", 1, 60, 30)

    U0 = u_in * 0.46
    y0 = [U0, 0, 0, 0, 0, 0, 0, 0, 0]
    p = [ku_in, 0.5, 0.005, 0.01, 0.02, 0.008, 0.012, 0.02, 0.01, th_in]

    t = np.linspace(0, days * 24, 1000)
    res = odeint(nitrogen_model, y0, t, args=(p,))
    
    def calc_sens(y_start, params_base, index, d=0.01):
        s1 = odeint(nitrogen_model, y_start, t, args=(params_base,))[-1][5]
        p_new = list(params_base)
        p_new[index] *= (1 + d)
        s2 = odeint(nitrogen_model, y_start, t, args=(tuple(p_new),))[-1][5]
        return ((s2 - s1) / s1) / d if s1 != 0 else 0

    elas_th = calc_sens(y0, p, 9)
    elas_ku = calc_sens(y0, p, 0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t/24, res[:,0], label="Urea", color='teal')
    ax.plot(t/24, res[:,2], label="Ammonium", color='royalblue')
    ax.plot(t/24, res[:,4], label="Nitrate", color='crimson')
    ax.plot(t/24, res[:,5], label="Plant N", color='green', lw=3)
    ax.set_xlabel("Days")
    ax.set_ylabel("kg N / ha")
    ax.legend(loc='upper right')
    st.pyplot(fig)

    st.subheader("Live Sensitivity & Efficiency")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("NUE", f"{(res[-1,5]/U0)*100:.1f}%")
    col_b.metric("Moisture Elasticity", f"{elas_th:.3f}")
    col_c.metric("Kinetics Elasticity", f"{elas_ku:.3f}")
    
    st.latex(r"S_{moisture} = \frac{\partial N_{plant}}{\partial \theta} \cdot \frac{\theta}{N_{plant}}")

with tab2:
    st.title("Inverse Regression Mode")
    st.write("Input your observed field Nitrogen values, and the model will predict the environmental conditions (θ and Urea) required to reach them.")
    
    target_plant = st.number_input("Observed Plant N (kg/ha)", value=50.0)
    target_loss = st.number_input("Observed Total Loss (kg/ha)", value=20.0)
    
    if st.button("Predict Environmental Parameters"):
        def fit_func(x):
            u_try, th_try = x
            u0_try = u_try * 0.46
            p_try = [0.07, 0.5, 0.005, 0.01, 0.02, 0.008, 0.012, 0.02, 0.01, th_try]
            sol_try = odeint(nitrogen_model, [u0_try,0,0,0,0,0,0,0,0], t, args=(p_try,))[-1]
            p_out = sol_try[5]
            l_out = sol_try[6] + sol_try[7] + sol_try[8]
            return (p_out - target_plant)**2 + (l_out - target_loss)**2

        opt = minimize(fit_func, x0=[200, 0.3], bounds=[(0, 600), (0.01, 0.99)])
        u_pred, th_pred = opt.x
        
        st.success(f"Predicted Urea Applied: {u_pred:.1f} kg/ha")
        st.success(f"Predicted Soil Moisture (θ): {th_pred:.2f}")
        
        u0_final = u_pred * 0.46
        p_final = [0.07, 0.5, 0.005, 0.01, 0.02, 0.008, 0.012, 0.02, 0.01, th_pred]
        res_f = odeint(nitrogen_model, [u0_final,0,0,0,0,0,0,0,0], t, args=(p_final,))
        
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(t/24, res_f[:,5], color='green', label="Predicted Plant N")
        ax2.plot(t/24, res_f[:,6]+res_f[:,7]+res_f[:,8], color='black', linestyle='--', label="Predicted Losses")
        ax2.set_title("Predicted System Trajectory")
        ax2.legend()
        st.pyplot(fig2)

st.write("---")
st.caption("Nitrogen Cycle Dynamics v2.0 | Normalized Sensitivity Analysis Included")
