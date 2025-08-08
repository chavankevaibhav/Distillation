import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Distillation Column Design Tool",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
R_gas = 8.314  # J/mol·K

# All the functions from your original code
def calculate_relative_volatility(bp1, bp2):
    Tb1 = bp1 + 273.15
    Tb2 = bp2 + 273.15
    dH_vap1 = 88 * Tb1  # Trouton's rule (J/mol)
    dH_vap2 = 88 * Tb2  # Trouton's rule (J/mol)
    T_avg = (Tb1 + Tb2) / 2
    alpha = np.exp((dH_vap1 - dH_vap2) / (R_gas * T_avg) * (Tb1 - Tb2) / T_avg)
    return alpha

def calculate_material_balance(feed_comp, xb, xd, feed_rate, mw1, mw2):
    mw_feed = feed_comp * mw1 + (1 - feed_comp) * mw2
    F_mol = feed_rate * 1000 / mw_feed  # kg/hr -> mol/hr
    D = F_mol * (feed_comp - xb) / (xd - xb)  # Distillate flow (mol/hr)
    B = F_mol - D  # Bottoms flow (mol/hr)
    return D, B, F_mol

def calculate_minimum_reflux_ratio(alpha, feed_comp, q_value):
    def underwood_eq(theta):
        return alpha * feed_comp / (alpha - theta) + (1 - feed_comp) / (1 - theta) - (1 - q_value)

    try:
        theta_sol = root_scalar(underwood_eq, bracket=[1.0001, alpha-0.0001]).root
        Rmin = (alpha * feed_comp) / (alpha - theta_sol) + (1 - feed_comp) / (1 - theta_sol) - 1
    except:
        Rmin = 1 / (alpha - 1) * (feed_comp / (feed_comp - alpha * (1 - feed_comp)))

    Rmin = max(Rmin, 0.1)
    return Rmin

def calculate_actual_reflux_ratio(Rmin, rr_mult):
    return rr_mult * Rmin

def calculate_minimum_stages(xd, xb, alpha):
    return np.log((xd/(1-xd)) * ((1-xb)/xb)) / np.log(alpha)

def calculate_actual_stages(Nmin, R, Rmin):
    X = (R - Rmin) / (R + 1)
    Y = 0.75 * (1 - X**0.5668)
    return (Nmin + Y) / (1 - Y)

def _equilibrium(x, alpha):
    return alpha * x / (1 + (alpha - 1) * x)

def _rectifying(x, R, xd):
    return R / (R + 1) * x + xd / (R + 1)

def _stripping(x, R, q, F, D, B, xb):
    Lbar = R * D + q * F
    Vbar = Lbar - B
    return Lbar / Vbar * x - B * xb / Vbar

def _q_line(x, q, zf):
    """Returns y on the q-line for a given x."""
    if abs(q - 1.0) < 1e-6: # Handle q = 1 (saturated liquid)
        return np.full_like(x, np.nan) # Return NaN for a vertical line
    else:
        return q / (q - 1) * x - zf / (q - 1)

def calculate_stages(alpha, R, xd, xb, feed_comp, q_value, F_mol, D, B):
    def equilibrium(x):
        return alpha * x / (1 + (alpha - 1) * x)

    def rectifying_line(x):
        return (R/(R+1)) * x + xd/(R+1)

    def stripping_line(x):
        L_bar = R * D + q_value * F_mol
        V_bar = L_bar - B
        return (L_bar / V_bar) * x - (B * xb) / V_bar

    x_stages = [xd]
    y_stages = [xd]
    x_current = xd
    stage = 0

    while x_current > xb:
        y_eq = equilibrium(x_current)
        x_stages.append(x_current)
        y_stages.append(y_eq)

        if stage == 0 or (y_eq > stripping_line(x_current) and feed_comp < x_current):
            y_next = rectifying_line(x_current) if y_stages[-1] > q_value else stripping_line(x_current)
        else:
            y_next = stripping_line(x_current)

        # Define L_bar and V_bar before using them
        L_bar = R * D + q_value * F_mol
        V_bar = L_bar - B

        x_next = (y_next - xd/(R+1)) * (R+1)/R if y_next > q_value else (y_next + (B*xb)/V_bar) * V_bar/L_bar

        x_stages.append(x_next)
        y_stages.append(y_next)
        x_current = x_next
        stage += 1

        if stage > 100:  # Safety break
            break

    return x_stages, y_stages, stage

def plot_mccabe_thiele(alpha, R, xd, xb, zf, q, F, D, B):
    """
    Draw a fully labelled McCabe-Thiele diagram and return the figure.
    """
    x = np.linspace(0, 1, 400)
    y_eq = _equilibrium(x, alpha)

    y_rect = _rectifying(x, R, xd)
    y_strip = _stripping(x, R, q, F, D, B, xb)

    qx = np.linspace(0, 1, 2)
    qy = _q_line(qx, q, zf)

    stairs_x, stairs_y = [xd], [xd]
    feed_found = False
    feed_stage = None
    n_stages = 0

    x_op, y_op = xd, xd

    while x_op > xb and n_stages < 100:
        y_eq_pt = _equilibrium(x_op, alpha)
        stairs_x.extend([x_op, x_op])
        stairs_y.extend([y_op, y_eq_pt])

        q_pt = _q_line(x_op, q, zf)
        if y_eq_pt > q_pt and not feed_found:
            x_next = (y_eq_pt - xd/(R+1)) * (R+1)/R
            y_next = _rectifying(x_next, R, xd)
        else:
            if not feed_found:
                feed_found = True
                feed_stage = n_stages + 1
            Lbar = R * D + q * F
            Vbar = Lbar - B
            x_next = (y_eq_pt + B * xb / Vbar) * Vbar / Lbar
            y_next = _stripping(x_next, R, q, F, D, B, xb)

        stairs_x.extend([x_next, x_next])
        stairs_y.extend([y_eq_pt, y_next])

        x_op, y_op = x_next, y_next
        n_stages += 1
        if x_op < xb:
            break

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x, y_eq, 'b', lw=2, label='Equilibrium')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.7, alpha=0.5)

    ax.plot(x, y_rect, 'g', lw=2, label='Rectifying')
    ax.plot(x, y_strip, 'r', lw=2, label='Stripping')

    if abs(q - 1.0) < 1e-6:
        ax.axvline(x=zf, color='m', linestyle='--', lw=2, label='q-line')
    elif abs(q - 1.0) > 1e-3:
        ax.plot(qx, qy, 'm--', lw=2, label='q-line')

    ax.plot(stairs_x, stairs_y, 'o-', color='orange', markersize=4,
            label='Stages', alpha=0.8)

    if feed_stage is not None:
        idx = 2 * feed_stage + 1
        if idx < len(stairs_x):
            ax.scatter(stairs_x[idx], stairs_y[idx], color='purple',
                       s=100, zorder=5, label=f'Feed stage ({feed_stage})')

    for i in range(1, min(n_stages + 1, 20)):  # Limit labels to avoid clutter
        idx = 4 * i - 2
        if idx < len(stairs_x):
            ax.text(stairs_x[idx] + 0.01, stairs_y[idx], str(i),
                    fontsize=8, color='navy', alpha=0.8)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Liquid mole fraction (light component)', fontsize=12)
    ax.set_ylabel('Vapor mole fraction (light component)', fontsize=12)
    ax.set_title('McCabe-Thiele Diagram', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    return fig, n_stages, feed_stage

def calculate_energy_and_cost(D, R, feed_comp, q_value, F_mol, mw1, mw2, xd, xb, n_stages, bp1, bp2, energy_cost_per_kwh, tower_cost_mult, condenser_cost_mult, reboiler_cost_mult):
    """
    Calculates energy consumption and equipment costs with customizable cost parameters.
    """
    avg_dH_vap = feed_comp * 88 * (bp1 + 273.15) + (1 - feed_comp) * 88 * (bp2 + 273.15)  # J/mol
    V = (R + 1) * D  # Vapor flow (mol/hr)
    Q_cond = V * avg_dH_vap / 3.6e6  # kWh
    Q_reb = (V + (1 - q_value) * F_mol) * avg_dH_vap / 3.6e6  # kWh

    tower_cost = tower_cost_mult * 15000 * (n_stages ** 0.8)
    condenser_cost = condenser_cost_mult * 5000 * (Q_cond ** 0.65)
    reboiler_cost = reboiler_cost_mult * 6000 * (Q_reb ** 0.7)
    total_equip_cost = tower_cost + condenser_cost + reboiler_cost
    energy_cost_hr = (Q_cond + Q_reb) * energy_cost_per_kwh  # $/kWh

    distillate_kg = D * (xd * mw1 + (1 - xd) * mw2) / 1000  # kg/hr
    cost_per_kg = energy_cost_hr / distillate_kg if distillate_kg > 0 else 0

    return Q_cond, Q_reb, total_equip_cost, energy_cost_hr, cost_per_kg, tower_cost, condenser_cost, reboiler_cost

# Streamlit App
def main():
    st.title("⚗️ Distillation Column Design Tool")
    st.markdown("### Design and analyze binary distillation columns with McCabe-Thiele method")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📋 Input Parameters")
        
        # Component properties
        st.subheader("Component Properties")
        col1, col2 = st.columns(2)
        with col1:
            bp1 = st.number_input("Light component BP (°C)", value=78.4, step=0.1, help="Boiling point of light component")
            mw1 = st.number_input("Light component MW (g/mol)", value=46.07, step=0.01, help="Molecular weight of light component")
        with col2:
            bp2 = st.number_input("Heavy component BP (°C)", value=100.0, step=0.1, help="Boiling point of heavy component")
            mw2 = st.number_input("Heavy component MW (g/mol)", value=18.02, step=0.01, help="Molecular weight of heavy component")
        
        # Operating conditions
        st.subheader("Operating Conditions")
        feed_comp = st.slider("Feed composition (light key)", 0.01, 0.99, 0.5, 0.01, help="Mole fraction of light component in feed")
        xd = st.slider("Distillate composition", 0.01, 0.99, 0.95, 0.01, help="Mole fraction of light component in distillate")
        xb = st.slider("Bottoms composition", 0.01, 0.99, 0.05, 0.01, help="Mole fraction of light component in bottoms")
        
        feed_rate = st.number_input("Feed rate (kg/hr)", value=1000.0, min_value=1.0, step=10.0)
        q_value = st.slider("q-value", 0.0, 2.0, 1.0, 0.1, help="0=saturated vapor, 1=saturated liquid, >1=subcooled liquid")
        rr_mult = st.slider("Reflux ratio multiplier", 1.1, 5.0, 1.5, 0.1, help="Multiple of minimum reflux ratio")
        
        # Cost parameters
        st.subheader("Economic Parameters")
        energy_cost = st.number_input("Energy cost ($/kWh)", value=0.10, step=0.01, format="%.3f")
        tower_mult = st.slider("Tower cost multiplier", 0.5, 2.0, 1.0, 0.1)
        condenser_mult = st.slider("Condenser cost multiplier", 0.5, 2.0, 1.0, 0.1)
        reboiler_mult = st.slider("Reboiler cost multiplier", 0.5, 2.0, 1.0, 0.1)
    
    # Main content area
    if st.button("🔬 Calculate Design", type="primary"):
        try:
            # Calculations
            alpha = calculate_relative_volatility(bp1, bp2)
            D, B, F_mol = calculate_material_balance(feed_comp, xb, xd, feed_rate, mw1, mw2)
            Rmin = calculate_minimum_reflux_ratio(alpha, feed_comp, q_value)
            R = calculate_actual_reflux_ratio(Rmin, rr_mult)
            Nmin = calculate_minimum_stages(xd, xb, alpha)
            N_actual = calculate_actual_stages(Nmin, R, Rmin)
            
            # Stage-by-stage calculation
            x_stages, y_stages, n_stages = calculate_stages(alpha, R, xd, xb, feed_comp, q_value, F_mol, D, B)
            
            # Energy and cost calculations
            Q_cond, Q_reb, total_cost, energy_cost_hr, cost_per_kg, tower_cost, condenser_cost, reboiler_cost = calculate_energy_and_cost(
                D, R, feed_comp, q_value, F_mol, mw1, mw2, xd, xb, n_stages, bp1, bp2, 
                energy_cost, tower_mult, condenser_mult, reboiler_mult
            )
            
            # Display results in columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📊 McCabe-Thiele Diagram")
                fig, stages_graphical, feed_stage = plot_mccabe_thiele(alpha, R, xd, xb, feed_comp, q_value, F_mol, D, B)
                st.pyplot(fig)
            
            with col2:
                st.subheader("📈 Design Results")
                
                # Key results
                results_df = pd.DataFrame({
                    'Parameter': [
                        'Relative volatility (α)',
                        'Minimum reflux ratio',
                        'Actual reflux ratio', 
                        'Minimum stages',
                        'Actual stages (Gilliland)',
                        'Actual stages (McCabe-Thiele)',
                        'Feed stage'
                    ],
                    'Value': [
                        f"{alpha:.3f}",
                        f"{Rmin:.3f}",
                        f"{R:.3f}",
                        f"{Nmin:.1f}",
                        f"{N_actual:.1f}",
                        f"{stages_graphical}",
                        f"{feed_stage if feed_stage else 'N/A'}"
                    ]
                })
                
                st.dataframe(results_df, hide_index=True)
            
            # Material balance
            st.subheader("⚖️ Material Balance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Feed Rate", f"{F_mol:.1f} mol/hr")
            with col2:
                st.metric("Distillate Rate", f"{D:.1f} mol/hr")
            with col3:
                st.metric("Bottoms Rate", f"{B:.1f} mol/hr")
            
            # Energy requirements
            st.subheader("⚡ Energy Requirements")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Condenser Duty", f"{Q_cond:.1f} kW")
            with col2:
                st.metric("Reboiler Duty", f"{Q_reb:.1f} kW")
            with col3:
                st.metric("Energy Cost", f"${energy_cost_hr:.2f}/hr")
            
            # Cost breakdown
            st.subheader("💰 Equipment Costs")
            cost_data = {
                'Equipment': ['Tower', 'Condenser', 'Reboiler', 'Total'],
                'Cost ($)': [f"{tower_cost:,.0f}", f"{condenser_cost:,.0f}", f"{reboiler_cost:,.0f}", f"{total_cost:,.0f}"]
            }
            cost_df = pd.DataFrame(cost_data)
            st.dataframe(cost_df, hide_index=True)
            
            # Additional metrics
            st.subheader("📋 Production Metrics")
            distillate_kg = D * (xd * mw1 + (1 - xd) * mw2) / 1000
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Distillate Production", f"{distillate_kg:.1f} kg/hr")
            with col2:
                st.metric("Cost per kg Distillate", f"${cost_per_kg:.3f}/kg")
                
        except Exception as e:
            st.error(f"Calculation error: {str(e)}")
            st.info("Please check your input parameters and try again.")
    
    # Information section
    with st.expander("ℹ️ About this tool"):
        st.markdown("""
        This tool designs binary distillation columns using the McCabe-Thiele graphical method.
        
        **Key Features:**
        - Calculates minimum reflux ratio and stages
        - Generates McCabe-Thiele diagram with stage stepping
        - Estimates energy requirements and equipment costs
        - Provides complete material balance
        
        **Assumptions:**
        - Binary system with constant relative volatility
        - Constant molal overflow
        - Trouton's rule for enthalpy of vaporization (ΔHvap = 88×Tb)
        - Ideal vapor-liquid equilibrium
        
        **Notes:**
        - Light component should have lower boiling point
        - q = 1.0 for saturated liquid feed
        - Reflux ratio multiplier typically 1.2-2.0 times minimum
        """)

if __name__ == "__main__":
    main()
