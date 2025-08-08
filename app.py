
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import io

# Import functions from the Python file
from distillation_funcs import (
    calculate_relative_volatility, calculate_material_balance,
    calculate_minimum_reflux_ratio, calculate_actual_reflux_ratio,
    calculate_minimum_stages, calculate_actual_stages,
    calculate_energy_and_cost, plot_mccabe_thiele, calculate_feed_stage,
    fetch_nist_data, antoine_vapor_pressure # Import antoine function for potential use
)

# Predefined components with boiling points (°C), molecular weights (g/mol), and Antoine Coefficients (A, B, C for kPa, °C)
# Note: These are example Antoine coefficients. Real values should be sourced from databases.
predefined_components = {
    "Ethanol": {"bp": 78.4, "mw": 46.07, "antoine": (16.8958, 3795.17, 230.918)}, # Example coeffs for kPa, C
    "Water": {"bp": 100.0, "mw": 18.02, "antoine": (16.3872, 3885.70, 230.170)}, # Example coeffs for kPa, C
    "Methanol": {"bp": 64.7, "mw": 32.04, "antoine": (16.5789, 3638.27, 239.500)}, # Example coeffs for kPa, C
    "Propanol": {"bp": 97.2, "mw": 60.10, "antoine": (16.7836, 4067.19, 213.410)}, # Example coeffs for kPa, C
    "Benzene": {"bp": 80.1, "mw": 78.11, "antoine": (15.9008, 2788.51, 220.790)}, # Example coeffs for kPa, C
    "Toluene": {"bp": 110.6, "mw": 92.14, "antoine": (16.0137, 3096.52, 219.480)}, # Example coeffs for kPa, C
}

# Streamlit App
st.title("Distillation Column Simulation (Rigorous VLE)")
st.write("Simulate the performance and economics of a binary distillation column using rigorous VLE calculations.")

st.sidebar.header("Input Parameters")

# Component Selection
component_options = list(predefined_components.keys())
selected_comp1_name = st.sidebar.selectbox("Select Component 1 (Light Key) or Enter Custom", component_options + ["Enter Custom"])

comp1 = ""
bp1 = None
mw1 = None
antoine_coeffs1 = None

if selected_comp1_name == "Enter Custom":
    custom_comp1_name = st.sidebar.text_input("Enter Custom Component 1 Name")
    if custom_comp1_name:
        # Attempt to fetch data from NIST (if fetch_nist_data is implemented to get Antoine)
        # For now, assume manual input for custom components
        st.sidebar.warning(f"Fetching Antoine coefficients from NIST is not yet fully implemented. Please enter properties manually for {custom_comp1_name}.")
        comp1 = custom_comp1_name
        bp1 = st.sidebar.number_input(f"Boiling point of {comp1} (°C)", key=f"{comp1}_bp", format="%.1f")
        mw1 = st.sidebar.number_input(f"Molecular weight of {comp1} (g/mol)", key=f"{comp1}_mw", format="%.2f")
        st.sidebar.subheader(f"Antoine Coefficients for {comp1} (kPa, °C)")
        A1 = st.sidebar.number_input("A1", key=f"{comp1}_A", format="%.4f")
        B1 = st.sidebar.number_input("B1", key=f"{comp1}_B", format="%.4f")
        C1 = st.sidebar.number_input("C1", key=f"{comp1}_C", format="%.4f")
        antoine_coeffs1 = (A1, B1, C1)

    else:
        comp1 = "Component 1"
        bp1 = st.sidebar.number_input(f"Boiling point of {comp1} (°C)", key=f"{comp1}_bp_default", format="%.1f")
        mw1 = st.sidebar.number_input(f"Molecular weight of {comp1} (g/mol)", key=f"{comp1}_mw_default", format="%.2f")
        st.sidebar.subheader(f"Antoine Coefficients for {comp1} (kPa, °C)")
        A1 = st.sidebar.number_input("A1", key=f"{comp1}_A_default", format="%.4f")
        B1 = st.sidebar.number_input("B1", key=f"{comp1}_B_default", format="%.4f")
        C1 = st.sidebar.number_input("C1", key=f"{comp1}_C_default", format="%.4f")
        antoine_coeffs1 = (A1, B1, C1)

else:
    comp1 = selected_comp1_name
    bp1 = predefined_components[comp1]["bp"]
    mw1 = predefined_components[comp1]["mw"]
    antoine_coeffs1 = predefined_components[comp1]["antoine"]


component_options_comp2 = [comp for comp in component_options if comp != comp1 and comp != "Enter Custom"]
selected_comp2_name = st.sidebar.selectbox("Select Component 2 (Heavy Key) or Enter Custom", component_options_comp2 + ["Enter Custom"])

comp2 = ""
bp2 = None
mw2 = None
antoine_coeffs2 = None

if selected_comp2_name == "Enter Custom":
    custom_comp2_name = st.sidebar.text_input("Enter Custom Component 2 Name")
    if custom_comp2_name:
        # Attempt to fetch data from NIST
        st.sidebar.warning(f"Fetching Antoine coefficients from NIST is not yet fully implemented. Please enter properties manually for {custom_comp2_name}.")
        comp2 = custom_comp2_name
        bp2 = st.sidebar.number_input(f"Boiling point of {comp2} (°C)", key=f"{comp2}_bp", format="%.1f")
        mw2 = st.sidebar.number_input(f"Molecular weight of {comp2} (g/mol)", key=f"{comp2}_mw", format="%.2f")
        st.sidebar.subheader(f"Antoine Coefficients for {comp2} (kPa, °C)")
        A2 = st.sidebar.number_input("A2", key=f"{comp2}_A", format="%.4f")
        B2 = st.sidebar.number_input("B2", key=f"{comp2}_B", format="%.4f")
        C2 = st.sidebar.number_input("C2", key=f"{comp2}_C", format="%.4f")
        antoine_coeffs2 = (A2, B2, C2)
    else:
        comp2 = "Component 2"
        bp2 = st.sidebar.number_input(f"Boiling point of {comp2} (°C)", key=f"{comp2}_bp_default", format="%.1f")
        mw2 = st.sidebar.number_input(f"Molecular weight of {comp2} (g/mol)", key=f"{comp2}_mw_default", format="%.2f")
        st.sidebar.subheader(f"Antoine Coefficients for {comp2} (kPa, °C)")
        A2 = st.sidebar.number_input("A2", key=f"{comp2}_A_default", format="%.4f")
        B2 = st.sidebar.number_input("B2", key=f"{comp2}_B_default", format="%.4f")
        C2 = st.sidebar.number_input("C2", key=f"{comp2}_C_default", format="%.4f")
        antoine_coeffs2 = (A2, B2, C2)

else:
    comp2 = selected_comp2_name
    bp2 = predefined_components[comp2]["bp"]
    mw2 = predefined_components[comp2]["mw"]
    antoine_coeffs2 = predefined_components[comp2]["antoine"]


# Display selected component properties and Antoine coefficients
st.sidebar.write(f"**{comp1} Properties:**")
st.sidebar.write(f"Boiling point: {bp1}°C")
st.sidebar.write(f"Molecular weight: {mw1} g/mol")
st.sidebar.write(f"Antoine Coeffs (A, B, C): {antoine_coeffs1}")
st.sidebar.write(f"**{comp2} Properties:**")
st.sidebar.write(f"Boiling point: {bp2}°C")
st.sidebar.write(f"Molecular weight: {mw2} g/mol")
st.sidebar.write(f"Antoine Coeffs (A, B, C): {antoine_coeffs2}")


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

st.sidebar.header("Activity Coefficient Model (Placeholder)")
activity_model = st.sidebar.selectbox("Select Activity Coefficient Model", ["Ideal (gamma=1.0)", "NRTL (Placeholder)", "UNIQUAC (Placeholder)"])

# Placeholder inputs for activity coefficient parameters
gamma1 = 1.0
gamma2 = 1.0
if activity_model != "Ideal (gamma=1.0)":
    st.sidebar.warning(f"{activity_model} model is a placeholder. Activity coefficients are currently fixed at 1.0.")
    # Add placeholder inputs for model parameters if needed in the future
    # nrtl_params = (st.sidebar.number_input("NRTL alpha12", value=0.3), st.sidebar.number_input("NRTL tau12", value=1.0), st.sidebar.number_input("NRTL tau21", value=1.0))
    # uniquac_params = (...)
    pass # Keep gamma1 and gamma2 as 1.0 for now


st.sidebar.header("Cost Parameters")
energy_cost_per_kwh = st.sidebar.number_input("Energy Cost ($/kWh)", value=0.10, format="%.2f")
tower_cost_mult = st.sidebar.number_input("Column Cost Multiplier", value=1.0, format="%.2f")
condenser_cost_mult = st.sidebar.number_input("Condenser Cost Multiplier", value=1.0, format="%.2f")
reboiler_cost_mult = st.sidebar.number_input("Reboiler Cost Multiplier", value=1.0, format="%.2f")


if st.button("Run Simulation"):
    # Add a check to ensure xd > feed_comp > xb and bp1 < bp2
    if not (xd > feed_comp > xb):
        st.error(f"Distillate composition ({xd}) must be greater than feed composition ({feed_comp}), and feed composition must be greater than bottoms composition ({xb}). Please adjust the input parameters.")
    elif not (bp1 is not None and bp2 is not None and bp1 < bp2):
         st.error(f"Boiling point of Component 1 ({bp1}°C) must be less than the boiling point of Component 2 ({bp2}°C) for Component 1 to be the light key. Please adjust the components or their boiling points.")
    elif antoine_coeffs1 is None or antoine_coeffs2 is None or any(np.isnan(antoine_coeffs1)) or any(np.isnan(antoine_coeffs2)):
         st.error("Antoine coefficients must be provided for both components.")
    else:
        # Calculate alpha based on Antoine coefficients at average temperature/pressure if needed for Rmin?
        # Rmin calculation uses alpha and q. alpha is typically relative volatility at feed conditions.
        # For rigorous VLE, alpha is not constant. The Rmin calculation based on Underwood equation
        # is more complex and depends on the VLE at the feed stage.
        # Let's use the simplified Rmin calculation for now, acknowledging this is an approximation
        # in the context of rigorous VLE. A truly rigorous Rmin requires solving Underwood's roots.
        # For a simple approach, we could calculate alpha at the feed temperature/pressure or average conditions.
        # Let's stick to the original Rmin calculation using an 'average' alpha based on boiling points for now,
        # as the rigorous VLE is primarily for the McCabe-Thiele plot and stage stepping.
        # A more rigorous Rmin calculation would be a future enhancement.
        alpha = calculate_relative_volatility(bp1, bp2) # Still using simplified alpha for Rmin for now


        if alpha <= 1:
             st.error(f"Calculated Relative Volatility (α) is {alpha:.4f}. It must be greater than 1 for separation to be possible. Please adjust the components.")
        else:

            D, B, F_mol = calculate_material_balance(feed_comp, xb, xd, feed_rate, mw1, mw2)
            Rmin = calculate_minimum_reflux_ratio(alpha, feed_comp, q_value) # Using simplified Rmin

            # Ensure Rmin is not negative or extremely small before calculating R and N_gilliland
            if Rmin < 1e-3: # Use a small threshold instead of 0
                 st.warning(f"Calculated Minimum Reflux Ratio (Rmin) is very small ({Rmin:.2f}). Setting Rmin to 0.1 for calculation stability.")
                 Rmin = 0.1


            if Rmin < 0: # Should be caught by the previous check, but as a safeguard
                 st.error(f"Calculated Minimum Reflux Ratio (Rmin) is negative ({Rmin:.2f}). This indicates an issue with the input compositions or the relative volatility. Please ensure xd > feed_comp > xb and alpha > 1.")
            else:
                R = calculate_actual_reflux_ratio(Rmin, rr_mult)
                Nmin = calculate_minimum_stages(xd, xb, alpha) # Using simplified Nmin (Fenske)

                # Ensure R > Rmin for Gilliland correlation
                if R <= Rmin:
                    st.warning(f"Actual Reflux Ratio (R) ({R:.2f}) must be greater than Minimum Reflux Ratio (Rmin) ({Rmin:.2f}) for the Gilliland correlation. Actual Stages (Gilliland) will be infinite.")
                    N_gilliland = float('inf')
                else:
                     # Add a check for Nmin being positive before Gilliland
                     if Nmin <= 0:
                          st.warning(f"Calculated Minimum Stages (Nmin) is not positive ({Nmin:.1f}). Cannot calculate Actual Stages using Gilliland correlation. Setting Actual Stages (Gilliland) to infinity.")
                          N_gilliland = float('inf')
                     else:
                          N_gilliland = calculate_actual_stages(Nmin, R, Rmin)


                # Call the rigorous plotting function with Antoine coeffs and activity coeffs
                fig, n_stages_mccabe_thiele, feed_stage_mccabe_thiele = plot_mccabe_thiele(
                    antoine_coeffs1, antoine_coeffs2, R, xd, xb, feed_comp, q_value, F_mol, D, B, pressure, gamma1, gamma2
                )

                # Calculate energy and cost using the number of stages from McCabe-Thiele
                Q_cond, Q_reb, total_equip_cost, energy_cost_hr, cost_per_kg, tower_cost, condenser_cost, reboiler_cost = calculate_energy_and_cost(
                    D, R, feed_comp, q_value, F_mol, mw1, mw2, xd, xb, n_stages_mccabe_thiele, bp1, bp2, # Pass bp1, bp2 for Trouton's rule in energy calc
                    energy_cost_per_kwh, tower_cost_mult, condenser_cost_mult, reboiler_cost_mult
                )

                # Calculate the feed stage using the new function (which uses rigorous stepping)
                calculated_feed_stage = calculate_feed_stage(
                     antoine_coeffs1, antoine_coeffs2, R, xd, xb, feed_comp, q_value, F_mol, D, B, pressure, gamma1, gamma2
                )


                st.subheader("Simulation Results")

                # Prepare results for download
                results_text = f"""Distillation Simulation Results (Rigorous VLE)

Input Parameters:
Component 1 (Light Key): {comp1}
Boiling Point 1: {bp1}°C
Molecular Weight 1: {mw1} g/mol
Antoine Coeffs 1 (A, B, C - kPa, °C): {antoine_coeffs1}
Component 2 (Heavy Key): {comp2}
Boiling Point 2: {bp2}°C
Molecular Weight 2: {mw2} g/mol
Antoine Coeffs 2 (A, B, C - kPa, °C): {antoine_coeffs2}
Feed Mole Fraction ({comp1}): {feed_comp:.2f}
Feed Flow Rate: {feed_rate:.1f} kg/hr
Operating Pressure: {pressure:.1f} atm
Distillate Mole Fraction ({comp1}): {xd:.2f}
Bottoms Mole Fraction ({comp1}): {xb:.2f}
Reflux Ratio Multiplier (R/Rmin): {rr_mult:.2f}
Feed Thermal Condition: {feed_phase} (q = {q_value:.2f})
Activity Coefficient Model: {activity_model}
Energy Cost: ${energy_cost_per_kwh:.2f}/kWh
Column Cost Multiplier: {tower_cost_mult:.2f}
Condenser Cost Multiplier: {condenser_cost_mult:.2f}
Reboiler Cost Multiplier: {reboiler_cost_mult:.2f}

Distillation Parameters:
VLE Model: Antoine Equation + {activity_model}
Relative Volatility (α, simplified): {alpha:.4f} (For Rmin/Nmin)
Minimum Reflux Ratio (Rmin, simplified): {Rmin:.2f}
Actual Reflux Ratio (R): {R:.2f}
Minimum Stages (Nmin, Fenske): {Nmin:.1f}
Theoretical Stages (McCabe-Thiele, Rigorous VLE): {n_stages_mccabe_thiele:.0f}
Actual Stages (Gilliland Correlation): {'Infinite' if N_gilliland == float('inf') else f'{N_gilliland:.1f}'}

Material Balance:
Distillate Flow: {D:.1f} mol/hr ({D * (xd * mw1 + (1 - xd) * mw2) / 1000:.1f} kg/hr)
Bottoms Flow: {B:.1f} mol/hr ({B * (xb * mw1 + (1 - xb) * mw2) / 1000:.1f} kg/hr)
Feed Flow: {F_mol:.1f} mol/hr ({feed_rate:.1f} kg/hr)

Energy Consumption:
Condenser Duty: {Q_cond:.1f} kWh
Reboiler Duty: {Q_reb:.1f} kWh
Total Energy Cost/hr: ${energy_cost_hr:.2f}
Cost per kg of distillate: ${cost_per_kg:.4f}
Note: Vaporization enthalpy estimated using Trouton's rule based on normal boiling points.

Equipment Cost Estimate:
Column Cost: ${tower_cost:,.0f}
Condenser Cost: ${condenser_cost:,.0f}
Reboiler Cost: ${reboiler_cost:,.0f}
Total Equipment Cost: ${total_equip_cost:,.0f}

Feed Stage Estimation:
Estimated Feed Stage (McCabe-Thiele Plot Stepping): {feed_stage_mccabe_thiele} (From staircase)
Estimated Feed Stage (Calculated from Operating Lines Intersection): {calculated_feed_stage} (From intersection point)
"""

                with st.expander("Distillation Parameters"):
                    st.write("Key parameters describing the distillation process.")
                    st.write(f"**VLE Model:** Antoine Equation + {activity_model}")
                    st.write(f"**Relative Volatility (α, simplified):** {alpha:.4f} (Based on boiling points, used for simplified Rmin and Nmin calculations)")
                    st.write(f"**Minimum Reflux Ratio (Rmin, simplified):** {Rmin:.2f} (Calculated using the simplified α and Underwood equation)")
                    st.write(f"**Actual Reflux Ratio (R):** {R:.2f}")
                    st.write(f"**Minimum Stages (Nmin, Fenske):** {Nmin:.1f}")
                    st.write(f"**Theoretical Stages (McCabe-Thiele, Rigorous VLE):** {n_stages_mccabe_thiele:.0f}")

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
                    st.write("Note: Vaporization enthalpy is estimated using Trouton's rule based on normal boiling points for simplicity. A more rigorous energy balance would utilize enthalpies calculated from the chosen thermodynamic model.")
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
                     st.write(f"**Estimated Feed Stage (McCabe-Thiele Plot Stepping):** {feed_stage_mccabe_thiele} (Based on where the staircase intersects the operating line determined by the feed condition and reflux ratio)")
                     st.write(f"**Estimated Feed Stage (Calculated from Operating Lines Intersection):** {calculated_feed_stage} (Based on the calculated x-coordinate where the rectifying and stripping operating lines intersect)")


                st.subheader("McCabe-Thiele Diagram (Rigorous VLE)")
                st.pyplot(fig)
                st.write("The McCabe-Thiele diagram graphically represents the separation stages using rigorous VLE calculated with the Antoine equation and the selected activity coefficient model.")

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
                st.write("Based on the provided inputs and rigorous VLE calculations, the simulation provides the following key insights:")
                st.write(f"- The separation is governed by the Vapor-Liquid Equilibrium (VLE) of the {comp1}-{comp2} mixture, calculated using the Antoine equation for vapor pressures and the **{activity_model}** model for activity coefficients.")
                st.write(f"- The simplified relative volatility (α) of {alpha:.4f} is provided for reference but is **not** used in the rigorous VLE calculations for the McCabe-Thiele plot or stage stepping. It's primarily for the simplified Rmin and Nmin calculations.")
                st.write(f"- The simplified minimum reflux ratio (Rmin) is calculated as {Rmin:.2f} using the simplified α and Underwood equation. A truly rigorous Rmin calculation would be more complex and depend on the rigorous VLE at the feed stage.")
                st.write(f"- The actual reflux ratio (R) of {R:.2f} is used for the simulation and the McCabe-Thiele diagram construction.")
                st.write(f"- The minimum number of theoretical stages (Nmin) required at infinite reflux is {Nmin:.1f} (calculated using the simplified Fenske equation).")
                st.write(f"- Using the actual reflux ratio and **rigorous VLE**, the McCabe-Thiele diagram stepping estimates **{n_stages_mccabe_thiele:.0f} theoretical stages** required for the desired separation.")
                st.write(f"- The Gilliland correlation estimates **{N_gilliland:.1f} actual stages** based on the simplified Nmin and R/Rmin. Note that the Gilliland correlation is an empirical method and the number of stages from the rigorous McCabe-Thiele plot is generally preferred when rigorous VLE is used.")
                st.write(f"- The simulation estimates a distillate flow rate of {D * (xd * mw1 + (1 - xd) * mw2) / 1000:.1f} kg/hr and a bottoms flow rate of {B * (xb * mw1 + (1 - xb) * mw2) / 1000:.1f} kg/hr for a feed rate of {feed_rate:.1f} kg/hr.")
                st.write(f"- Energy consumption is primarily driven by the condenser duty ({Q_cond:.1f} kWh) and reboiler duty ({Q_reb:.1f} kWh). **Note that the vaporization enthalpy used in these energy calculations is estimated using Trouton's rule based on normal boiling points for simplicity. A more rigorous energy balance would utilize enthalpies calculated from the chosen thermodynamic model.** This translates to an estimated energy cost of ${energy_cost_hr:.2f} per hour.")
                st.write(f"- The preliminary equipment cost estimate for the column, condenser, and reboiler is approximately ${total_equip_cost:,.0f}.")
                st.write(f"The optimal feed stage location is estimated to be stage **{feed_stage_mccabe_thiele}** (from the McCabe-Thiele plot stepping) or stage **{calculated_feed_stage}** (calculated from the intersection of operating lines). Placing the feed at or near this stage is crucial for minimizing the total number of stages required for the separation.")
                st.write("Adjusting parameters like the reflux ratio multiplier, feed composition, product purities, operating pressure, and the thermodynamic model parameters (Antoine coefficients, activity coefficients) will significantly impact the VLE, the number of stages, energy consumption, and ultimately the overall cost of the distillation process.")

                st.markdown("---")
                st.write("Source code: [https://github.com/your_username/your_repository_name]") # Add your source code link here
