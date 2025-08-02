
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# Import functions from the Python file
from distillation_funcs import (
    calculate_relative_volatility, calculate_material_balance,
    calculate_minimum_reflux_ratio, calculate_actual_reflux_ratio,
    calculate_minimum_stages, calculate_actual_stages,
    calculate_energy_and_cost, plot_mccabe_thiele, calculate_feed_stage # Import the new function
)

# Streamlit App
st.title("Distillation Column Simulation")

st.sidebar.header("Input Parameters")

comp1 = st.sidebar.text_input("Component 1", "Ethanol")
comp2 = st.sidebar.text_input("Component 2", "Water")
bp1 = st.sidebar.number_input(f"Boiling point of {comp1} (°C)", value=78.4, format="%.1f")
bp2 = st.sidebar.number_input(f"Boiling point of {comp2} (°C)", value=100.0, format="%.1f")
mw1 = st.sidebar.number_input(f"Molecular weight of {comp1} (g/mol)", value=46.07, format="%.2f")
mw2 = st.sidebar.number_input(f"Molecular weight of {comp2} (g/mol)", value=18.02, format="%.2f")
feed_comp = st.sidebar.slider(f"Mole fraction of {comp1} in feed", 0.0, 1.0, 0.4, 0.01)
feed_rate = st.sidebar.number_input("Feed flow rate (kg/hr)", value=1000.0, format="%.1f")
feed_temp = st.sidebar.number_input("Feed temperature (°C)", value=85.0, format="%.1f")
pressure = st.sidebar.number_input("Operating pressure (atm)", value=1.0, format="%.1f")
xd = st.sidebar.slider(f"Mole fraction of {comp1} in distillate", 0.0, 1.0, 0.95, 0.01)
xb = st.sidebar.slider(f"Mole fraction of {comp1} in bottoms", 0.0, 1.0, 0.05, 0.01)
rr_mult = st.sidebar.slider("Reflux ratio multiplier (R/Rmin)", 1.0, 5.0, 1.5, 0.1)
q_value = st.sidebar.slider("Feed thermal condition (q)", 0.0, 1.0, 1.0, 0.1)


if st.button("Run Simulation"):
    alpha = calculate_relative_volatility(bp1, bp2)
    D, B, F_mol = calculate_material_balance(feed_comp, xb, xd, feed_rate, mw1, mw2)
    Rmin = calculate_minimum_reflux_ratio(alpha, feed_comp, q_value)
    R = calculate_actual_reflux_ratio(Rmin, rr_mult)
    Nmin = calculate_minimum_stages(xd, xb, alpha)
    N_gilliland = calculate_actual_stages(Nmin, R, Rmin)
    fig, n_stages_mccabe_thiele, feed_stage_mccabe_thiele = plot_mccabe_thiele(alpha, R, xd, xb, feed_comp, q_value, F_mol, D, B)
    Q_cond, Q_reb, total_equip_cost, energy_cost_hr, cost_per_kg = calculate_energy_and_cost(D, R, feed_comp, q_value, F_mol, mw1, mw2, xd, xb, n_stages_mccabe_thiele, bp1, bp2)

    # Calculate the feed stage using the new function
    calculated_feed_stage = calculate_feed_stage(alpha, R, xd, xb, feed_comp, q_value, F_mol, D, B)

    st.subheader("Distillation Parameters")
    st.write(f"Relative Volatility (α): {alpha:.4f}")
    st.write(f"Minimum Reflux Ratio (Rmin): {Rmin:.2f}")
    st.write(f"Actual Reflux Ratio (R): {R:.2f}")
    st.write(f"Minimum Stages (Nmin): {Nmin:.1f}")
    st.write(f"Theoretical Stages (McCabe-Thiele): {n_stages_mccabe_thiele:.0f}")
    st.write(f"Actual Stages (Gilliland): {N_gilliland:.1f}")

    st.subheader("Material Balance")
    st.write(f"Distillate Flow: {D:.1f} mol/hr ({D * (xd * mw1 + (1 - xd) * mw2) / 1000:.1f} kg/hr)")
    st.write(f"Bottoms Flow: {B:.1f} mol/hr ({B * (xb * mw1 + (1 - xb) * mw2) / 1000:.1f} kg/hr)")
    st.write(f"Feed Flow: {F_mol:.1f} mol/hr ({feed_rate:.1f} kg/hr)")

    st.subheader("Energy Consumption")
    st.write(f"Condenser Duty: {Q_cond:.1f} kWh")
    st.write(f"Reboiler Duty: {Q_reb:.1f} kWh")
    st.write(f"Total Energy Cost/hr: ${energy_cost_hr:.2f}")
    st.write(f"Cost per kg of distillate: ${cost_per_kg:.4f}")

    st.subheader("Equipment Cost Estimate")
    tower_cost = 15000 * (n_stages_mccabe_thiele ** 0.8)
    condenser_cost = 5000 * (Q_cond ** 0.65)
    reboiler_cost = 6000 * (Q_reb ** 0.7)
    st.write(f"Column Cost: ${tower_cost:,.0f}")
    st.write(f"Condenser Cost: ${condenser_cost:,.0f}")
    st.write(f"Reboiler Cost: ${reboiler_cost:,.0f}")
    st.write(f"Total Equipment Cost: ${total_equip_cost:,.0f}")


    st.subheader("McCabe-Thiele Diagram")
    st.pyplot(fig)
    st.write(f"Estimated Feed Stage (McCabe-Thiele Plot): {feed_stage_mccabe_thiele}")
    # Display the calculated feed stage
    st.write(f"Estimated Feed Stage (Calculated): {calculated_feed_stage}")
