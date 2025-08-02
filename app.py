import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import io # Import the io module
import requests # Import requests for potential future use in fetching data
from bs4 import BeautifulSoup # Import BeautifulSoup for parsing HTML

# Import functions from the Python file
from distillation_funcs import (
    calculate_relative_volatility, calculate_material_balance,
    calculate_minimum_reflux_ratio, calculate_actual_reflux_ratio,
    calculate_minimum_stages, calculate_actual_stages,
    calculate_energy_and_cost, plot_mccabe_thiele, calculate_feed_stage,
    fetch_nist_data # Import the new data fetching function
)

# Predefined components with boiling points (°C) and molecular weights (g/mol)
predefined_components = {
    "Ethanol": {"bp": 78.4, "mw": 46.07},
    "Water": {"bp": 100.0, "mw": 18.02},
    "Methanol": {"bp": 64.7, "mw": 32.04},
    "Propanol": {"bp": 97.2, "mw": 60.10},
    "Benzene": {"bp": 80.1, "mw": 78.11},
    "Toluene": {"bp": 110.6, "mw": 92.14},
}

# Streamlit App
st.title("Distillation Column Simulation")
st.write("Simulate the performance and economics of a binary distillation column.")

st.sidebar.header("Input Parameters")

# Component Selection
component_options = list(predefined_components.keys())
selected_comp1_name = st.sidebar.selectbox("Select Component 1 (Light Key) or Enter Custom", component_options + ["Enter Custom"])

if selected_comp1_name == "Enter Custom":
    custom_comp1_name = st.sidebar.text_input("Enter Custom Component 1 Name")
    if custom_comp1_name:
        comp1_data = fetch_nist_data(custom_comp1_name)
        if comp1_data:
            comp1 = custom_comp1_name
            bp1 = comp1_data["bp"]
            mw1 = comp1_data["mw"]
            st.sidebar.success(f"Data for {comp1} fetched from NIST.")
        else:
            st.sidebar.error(f"Could not fetch data for {custom_comp1_name} from NIST. Please enter properties manually.")
            comp1 = custom_comp1_name if custom_comp1_name else "Component 1"
            bp1 = st.sidebar.number_input(f"Boiling point of {comp1} (°C)", format="%.1f")
            mw1 = st.sidebar.number_input(f"Molecular weight of {comp1} (g/mol)", format="%.2f")
    else:
        comp1 = "Component 1"
        bp1 = st.sidebar.number_input(f"Boiling point of {comp1} (°C)", format="%.1f")
        mw1 = st.sidebar.number_input(f"Molecular weight of {comp1} (g/mol)", format="%.2f")
else:
    comp1 = selected_comp1_name
    bp1 = predefined_components[comp1]["bp"]
    mw1 = predefined_components[comp1]["mw"]


component_options_comp2 = [comp for comp in component_options if comp != comp1 and comp != "Enter Custom"] # Exclude selected_comp1 and "Enter Custom"
selected_comp2_name = st.sidebar.selectbox("Select Component 2 (Heavy Key) or Enter Custom", component_options_comp2 + ["Enter Custom"])

if selected_comp2_name == "Enter Custom":
    custom_comp2_name = st.sidebar.text_input("Enter Custom Component 2 Name")
    if custom_comp2_name:
        comp2_data = fetch_nist_data(custom_comp2_name)
        if comp2_data:
            comp2 = custom_comp2_name
            bp2 = comp2_data["bp"]
            mw2 = comp2_data["mw"]
            st.sidebar.success(f"Data for {comp2} fetched from NIST.")
        else:
            st.sidebar.error(f"Could not fetch data for {custom_comp2_name} from NIST. Please enter properties manually.")
            comp2 = custom_comp2_name if custom_comp2_name else "Component 2"
            bp2 = st.sidebar.number_input(f"Boiling point of {comp2} (°C)", format="%.1f")
            mw2 = st.sidebar.number_input(f"Molecular weight of {comp2} (g/mol)", format="%.2f")
    else:
        comp2 = "Component 2"
        bp2 = st.sidebar.number_input(f"Boiling point of {comp2} (°C)", format="%.1f")
        mw2 = st.sidebar.number_input(f"Molecular weight of {comp2} (g/mol)", format="%.2f")

else:
    comp2 = selected_comp2_name
    bp2 = predefined_components[comp2]["bp"]
    mw2 = predefined_components[comp2]["mw"]

# Display selected component properties
st.sidebar.write(f"**{comp1} Properties:**")
st.sidebar.write(f"Boiling point: {bp1}°C")
st.sidebar.write(f"Molecular weight: {mw1} g/mol")
st.sidebar.write(f"**{comp2} Properties:**")
st.sidebar.write(f"Boiling point: {bp2}°C")
st.sidebar.write(f"Molecular weight: {mw2} g/mol")


feed_comp = st.sidebar.slider(f"Mole fraction of {comp1} in feed (zf)", 0.0, 1.0, 0.4, 0.01)
feed_rate = st.sidebar.number_input("Feed flow rate (kg/hr)", value=1000.0, format="%.1f")
pressure = st.sidebar.number_input("Operating pressure (atm)", value=1.0, format="%.1f")
xd = st.sidebar.slider(f"Mole fraction of {comp1} in distillate (xd)", 0.0, 1.0, 0.95, 0.01)
xb = st.sidebar.slider(f"Mole fraction of {comp1} in bottoms (xb)", 0.0, 1.0, 0.05, 0.01)
rr_mult = st.sidebar.slider("Reflux ratio multiplier (R/Rmin)", 1.0, 5.0, 1.5, 0.1)

# Feed Phase Selection and q_value calculation
feed_phase = st.sidebar.radio(
    "Feed Thermal Condition",
    ('Saturated Liquid', 'Saturated Vapor', 'Subcooled Liquid', 'Superheated Vapor', 'Partially Vaporized')
)

q_value = 1.0 # Default for Saturated Liquid
if feed_phase == 'Saturated Vapor':
    q_value = 0.0
elif feed_phase == 'Subcooled Liquid':
    st.sidebar.write("For Subcooled Liquid, q > 1")
    feed_temp = st.sidebar.number_input("Feed Temperature (°C)", value=25.0, format="%.1f")
    st.sidebar.warning("Simplified q calculation for Subcooled Liquid. More rigorous thermodynamics needed for accuracy.")
    q_value = st.sidebar.number_input("Enter q value for Subcooled Liquid", value=1.1, format="%.2f")

elif feed_phase == 'Superheated Vapor':
    st.sidebar.write("For Superheated Vapor, q < 0")
    feed_temp = st.sidebar.number_input("Feed Temperature (°C)", value=150.0, format="%.1f")
    st.sidebar.warning("Simplified q calculation for Superheated Vapor. More rigorous thermodynamics needed for accuracy.")
    q_value = st.sidebar.number_input("Enter q value for Superheated Vapor", value=-0.1, format="%.2f")

elif feed_phase == 'Partially Vaporized':
    st.sidebar.write("For Partially Vaporized, 0 < q < 1")
    vapor_fraction = st.sidebar.slider("Vaporized fraction of feed", 0.01, 0.99, 0.5, 0.01)
    q_value = 1.0 - vapor_fraction

st.sidebar.header("Cost Parameters")
energy_cost_per_kwh = st.sidebar.number_input("Energy Cost ($/kWh)", value=0.10, format="%.2f")
tower_cost_mult = st.sidebar.number_input("Column Cost Multiplier", value=1.0, format="%.2f")
condenser_cost_mult = st.sidebar.number_input("Condenser Cost Multiplier", value=1.0, format="%.2f")
reboiler_cost_mult = st.sidebar.number_input("Reboiler Cost Multiplier", value=1.0, format="%.2f")


if st.button("Run Simulation"):
    # Add a check to ensure xd > feed_comp > xb and bp1 < bp2
    if not (xd > feed_comp > xb):
        st.error(f"Distillate composition ({xd}) must be greater than feed composition ({feed_comp}), and feed composition must be greater than bottoms composition ({xb}). Please adjust the input parameters.")
    elif not (bp1 < bp2):
         st.error(f"Boiling point of Component 1 ({bp1}°C) must be less than the boiling point of Component 2 ({bp2}°C) for Component 1 to be the light key. Please adjust the components or their boiling points.")
    else:
        alpha = calculate_relative_volatility(bp1, bp2)

        if alpha <= 1:
             st.error(f"Calculated Relative Volatility (α) is {alpha:.4f}. It must be greater than 1 for separation to be possible. Please adjust the components.")
        else:

            D, B, F_mol = calculate_material_balance(feed_comp, xb, xd, feed_rate, mw1, mw2)
            Rmin = calculate_minimum_reflux_ratio(alpha, feed_comp, q_value)

            if Rmin < 0:
                 st.error(f"Calculated Minimum Reflux Ratio (Rmin) is negative ({Rmin:.2f}). This might indicate an issue with the input compositions or the relative volatility. Please ensure xd > feed_comp > xb and alpha > 1.")
            else:
                R = calculate_actual_reflux_ratio(Rmin, rr_mult)
                Nmin = calculate_minimum_stages(xd, xb, alpha)

                # Ensure R > Rmin for Gilliland correlation
                if R <= Rmin:
                    st.warning(f"Actual Reflux Ratio (R) ({R:.2f}) must be greater than Minimum Reflux Ratio (Rmin) ({Rmin:.2f}) for the Gilliland correlation. Actual Stages (Gilliland) will be infinite.")
                    N_gilliland = float('inf')
                else:
                     N_gilliland = calculate_actual_stages(Nmin, R, Rmin)

                fig, n_stages_mccabe_thiele, feed_stage_mccabe_thiele = plot_mccabe_thiele(alpha, R, xd, xb, feed_comp, q_value, F_mol, D, B)
                Q_cond, Q_reb, total_equip_cost, energy_cost_hr, cost_per_kg, tower_cost, condenser_cost, reboiler_cost = calculate_energy_and_cost(
                    D, R, feed_comp, q_value, F_mol, mw1, mw2, xd, xb, n_stages_mccabe_thiele, bp1, bp2,
                    energy_cost_per_kwh, tower_cost_mult, condenser_cost_mult, reboiler_cost_mult # Pass new cost parameters
                )

                # Calculate the feed stage using the new function
                calculated_feed_stage = calculate_feed_stage(alpha, R, xd, xb, feed_comp, q_value, F_mol, D, B)

                st.subheader("Simulation Results")

                # Prepare results for download
                results_text = f"""Distillation Simulation Results

Input Parameters:
Component 1 (Light Key): {comp1}
Boiling Point 1: {bp1}°C
Molecular Weight 1: {mw1} g/mol
Component 2 (Heavy Key): {comp2}
Boiling Point 2: {bp2}°C
Molecular Weight 2: {mw2} g/mol
Feed Mole Fraction ({comp1}): {feed_comp:.2f}
Feed Flow Rate: {feed_rate:.1f} kg/hr
Operating Pressure: {pressure:.1f} atm
Distillate Mole Fraction ({comp1}): {xd:.2f}
Bottoms Mole Fraction ({comp1}): {xb:.2f}
Reflux Ratio Multiplier (R/Rmin): {rr_mult:.2f}
Feed Thermal Condition: {feed_phase} (q = {q_value:.2f})
Energy Cost: ${energy_cost_per_kwh:.2f}/kWh
Column Cost Multiplier: {tower_cost_mult:.2f}
Condenser Cost Multiplier: {condenser_cost_mult:.2f}
Reboiler Cost Multiplier: {reboiler_cost_mult:.2f}

Distillation Parameters:
Relative Volatility (α): {alpha:.4f}
Minimum Reflux Ratio (Rmin): {Rmin:.2f}
Actual Reflux Ratio (R): {R:.2f}
Minimum Stages (Nmin, Fenske): {Nmin:.1f}
Theoretical Stages (McCabe-Thiele): {n_stages_mccabe_thiele:.0f}
Actual Stages (Gilliland Correlation): {N_gilliland:.1f if N_gilliland != float('inf') else 'Infinite'}

Material Balance:
Distillate Flow: {D:.1f} mol/hr ({D * (xd * mw1 + (1 - xd) * mw2) / 1000:.1f} kg/hr)
Bottoms Flow: {B:.1f} mol/hr ({B * (xb * mw1 + (1 - xb) * mw2) / 1000:.1f} kg/hr)
Feed Flow: {F_mol:.1f} mol/hr ({feed_rate:.1f} kg/hr)

Energy Consumption:
Condenser Duty: {Q_cond:.1f} kWh
Reboiler Duty: {Q_reb:.1f} kWh
Total Energy Cost/hr: ${energy_cost_hr:.2f}
Cost per kg of distillate: ${cost_per_kg:.4f}

Equipment Cost Estimate:
Column Cost: ${tower_cost:,.0f}
Condenser Cost: ${condenser_cost:,.0f}
Reboiler Cost: ${reboiler_cost:,.0f}
Total Equipment Cost: ${total_equip_cost:,.0f}

Feed Stage Estimation:
Estimated Feed Stage (McCabe-Thiele Plot): {feed_stage_mccabe_thiele}
Estimated Feed Stage (Calculated from Operating Lines Intersection): {calculated_feed_stage}
"""

                with st.expander("Distillation Parameters"):
                    st.write("Key parameters describing the distillation process.")
                    st.write(f"**Relative Volatility (α):** {alpha:.4f}")
                    st.write(f"**Minimum Reflux Ratio (Rmin):** {Rmin:.2f}")
                    st.write(f"**Actual Reflux Ratio (R):** {R:.2f}")
                    st.write(f"**Minimum Stages (Nmin, Fenske):** {Nmin:.1f}")
                    st.write(f"**Theoretical Stages (McCabe-Thiele):** {n_stages_mccabe_thiele:.0f}")

                    if N_gilliland == float('inf'):
                         st.write("**Actual Stages (Gilliland Correlation):** Infinite (R <= Rmin)")
                    else:
                         st.write(f"**Actual Stages (Gilliland Correlation):** {N_gilliland:.1f}")

                with st.expander("Material Balance"):
                    st.write("Flow rates of the feed, distillate, and bottoms streams.")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Distillate Flow:** {D:.1f} mol/hr ({D * (xd * mw1 + (1 - xd) * mw2) / 1000:.1f} kg/hr)")
                    with col2:
                        st.write(f"**Bottoms Flow:** {B:.1f} mol/hr ({B * (xb * mw1 + (1 - xb) * mw2) / 1000:.1f} kg/hr)")
                    st.write(f"**Feed Flow:** {F_mol:.1f} mol/hr ({feed_rate:.1f} kg/hr)")
                    st.write("Note: Values are for the light key component.")


                with st.expander("Energy Consumption"):
                    st.write("Estimated energy duties for the condenser and reboiler, and associated energy costs based on the provided energy cost.")
                    col3, col4 = st.columns(2)
                    with col3:
                         st.write(f"**Condenser Duty:** {Q_cond:.1f} kWh")
                    with col4:
                         st.write(f"**Reboiler Duty:** {Q_reb:.1f} kWh")
                    st.write(f"**Total Energy Cost/hr:** ${energy_cost_hr:.2f}")
                    st.write(f"**Cost per kg of distillate:** ${cost_per_kg:.4f}")


                with st.expander("Equipment Cost Estimate"):
                    st.write("Preliminary capital cost estimate for the distillation column and heat exchangers based on provided multipliers.")
                    st.write(f"**Column Cost:** ${tower_cost:,.0f}")
                    st.write(f"**Condenser Cost:** ${condenser_cost:,.0f}")
                    st.write(f"**Reboiler Cost:** ${reboiler_cost:,.0f}")
                    st.write(f"**Total Equipment Cost:** ${total_equip_cost:,.0f}")

                with st.expander("Feed Stage Estimation"):
                     st.write("Estimation of the optimal feed stage location.")
                     st.write(f"**Estimated Feed Stage (McCabe-Thiele Plot):** {feed_stage_mccabe_thiele}")
                     st.write(f"**Estimated Feed Stage (Calculated from Operating Lines Intersection):** {calculated_feed_stage}")

                st.subheader("McCabe-Thiele Diagram")
                st.pyplot(fig)
                st.write("The McCabe-Thiele diagram graphically represents the separation stages.")

                # Download buttons
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button(
                        label="Download Results",
                        data=results_text,
                        file_name="distillation_results.txt",
                        mime="text/plain"
                    )

                with col_dl2:
                    # Save figure to BytesIO
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)

                    st.download_button(
                        label="Download McCabe-Thiele Plot",
                        data=buf,
                        file_name="mccabe_thiele_plot.png",
                        mime="image/png"
                    )


                st.markdown("---")
                st.subheader("Summary and Interpretation")
                st.write("Based on the provided inputs, the simulation provides the following key insights:")
                st.write(f"- The relative volatility (α) of {alpha:.4f} indicates the ease of separation between {comp1} and {comp2}. A higher alpha means an easier separation, requiring fewer stages or lower reflux.")
                st.write(f"- The minimum reflux ratio (Rmin) is calculated as {Rmin:.2f}. Operating at this minimum would require an infinite number of stages. The actual reflux ratio (R) of {R:.2f} is used for practical design.")
                st.write(f"- The minimum number of theoretical stages (Nmin) required for the desired separation at infinite reflux is {Nmin:.1f}. Using the actual reflux ratio, the McCabe-Thiele diagram estimates {n_stages_mccabe_thiele:.0f} theoretical stages, while the Gilliland correlation estimates {N_gilliland:.1f} actual stages.")
                st.write(f"- The simulation estimates a distillate flow rate of {D * (xd * mw1 + (1 - xd) * mw2) / 1000:.1f} kg/hr and a bottoms flow rate of {B * (xb * mw1 + (1 - xb) * mw2) / 1000:.1f} kg/hr for a feed rate of {feed_rate:.1f} kg/hr.")
                st.write(f"- Energy consumption is primarily driven by the condenser duty ({Q_cond:.1f} kWh) and reboiler duty ({Q_reb:.1f} kWh). This translates to an estimated energy cost of ${energy_cost_hr:.2f} per hour.")
                st.write(f"- The preliminary equipment cost estimate for the column, condenser, and reboiler is approximately ${total_equip_cost:,.0f}.")
                st.write(f"The optimal feed stage location is estimated to be stage {feed_stage_mccabe_thiele} (from the McCabe-Thiele plot) or stage {calculated_feed_stage} (calculated intersection). Placing the feed at or near this stage is crucial for minimizing the total number of stages required for the separation.")
                st.write("Adjusting parameters like the reflux ratio multiplier (affecting R), feed composition, and product purities will impact the number of stages, energy consumption, and ultimately the overall cost of the distillation process. Higher reflux ratios generally lead to fewer stages but higher energy costs.")

                st.markdown("---")
                st.write("Source code: [Link to your repository here]") # Add your source code link here
