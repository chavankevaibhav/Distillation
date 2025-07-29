import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, fsolve
from scipy.interpolate import interp1d
import os
from datetime import datetime
import streamlit as st

# Try to import thermo, but provide fallback if not available
try:
    from thermo import Mixture
    THERMO_AVAILABLE = True
except ImportError:
    THERMO_AVAILABLE = False
    # st.warning("⚠️ thermo library not available. Using ideal approximations.") # Use Streamlit for warnings
    print("⚠️ thermo library not available. Using ideal approximations.")

try:
    from chemicals import MolarMass
    CHEMICALS_AVAILABLE = True
except ImportError:
    CHEMICALS_AVAILABLE = False

# Constants
R_gas = 8.314  # J/mol·K

# === VLE FROM NIST USING THERMO ===
def get_vle_data_nist(comp1, comp2, P=101325):  # Pressure in Pa
    """
    Fetch real VLE data using thermo (NIST/DIPPR)
    Returns: x_data, y_data, T_data (liquid, vapor, temperature)
    """
    if not THERMO_AVAILABLE:
        return None, None, None
        
    try:
        # Create mixture object (uses NIST data if available)
        mix = Mixture([comp1, comp2], zs=[0.5, 0.5], P=P)
        st.info(f"Using real VLE data for {comp1} + {comp2} at {P/101325:.2f} atm")

        # Generate VLE points
        x_vals = np.linspace(0.01, 0.99, 20)
        x_data, y_data, T_data = [], [], []

        for x in x_vals:
            try:
                m = Mixture([comp1, comp2], zs=[x, 1-x], P=P)
                # Check if mixture exists and has valid properties
                if hasattr(m, 'T') and m.T > 0 and hasattr(m, 'K') and len(m.K) >= 1:
                    x_data.append(x)
                    y_calc = m.K[0] * x if m.K[0] * x <= 1.0 else 1.0  # Ensure y <= 1
                    y_data.append(y_calc)
                    T_data.append(m.T - 273.15)  # °C
            except Exception as e:
                continue

        # Add pure components if possible
        try:
            m_pure1 = Mixture([comp1], P=P)
            if hasattr(m_pure1, 'T') and m_pure1.T > 0:
                x_data.insert(0, 1.0)
                y_data.insert(0, 1.0)
                T_data.insert(0, m_pure1.T - 273.15)
        except:
            pass
            
        try:
            m_pure2 = Mixture([comp2], P=P)
            if hasattr(m_pure2, 'T') and m_pure2.T > 0:
                x_data.append(0.0)
                y_data.append(0.0)
                T_data.append(m_pure2.T - 273.15)
        except:
            pass

        if len(x_data) < 3:  # Need at least 3 points for interpolation
            return None, None, None

        # Sort by x and remove duplicates
        combined = list(zip(x_data, y_data, T_data))
        # Remove duplicates based on x values
        seen_x = set()
        unique_data = []
        for x, y, t in combined:
            if x not in seen_x:
                seen_x.add(x)
                unique_data.append((x, y, t))
            
        sorted_data = sorted(unique_data, key=lambda item: item[0])
        x_data, y_data, T_data = zip(*sorted_data)

        return np.array(x_data), np.array(y_data), np.array(T_data)

    except Exception as e:
        st.warning(f"⚠️ NIST data not available: {e}. Using ideal approximation.")
        return None, None, None


# === IDEAL RELATIVE VOLATILITY (FALLBACK) ===
def calculate_relative_volatility_ideal(bp1, bp2):
    """Calculate relative volatility using boiling point difference"""
    Tb1 = bp1 + 273.15
    Tb2 = bp2 + 273.15
    # Trouton's rule approximation for enthalpy of vaporization
    dH_vap1 = 88 * Tb1  # J/mol
    dH_vap2 = 88 * Tb2  # J/mol
    T_avg = (Tb1 + Tb2) / 2
    
    # Approximate relative volatility
    if abs(Tb1 - Tb2) < 1:  # Very close boiling points
        alpha = 1.05
    else:
        alpha = np.exp((dH_vap1/Tb1 - dH_vap2/Tb2) / R_gas * (Tb1 - Tb2) / T_avg)
    
    return max(alpha, 1.05)


# === EQUILIBRIUM FROM INTERPOLATION OF NIST DATA ===
def create_equilibrium_function(x_vle, y_vle):
    """Create interpolation function for VLE data"""
    if len(x_vle) < 2:
        return lambda x: x  # Fallback to ideal case
        
    # Use linear interpolation if not enough points for cubic
    kind = 'cubic' if len(x_vle) > 3 else 'linear'
    f_eq = interp1d(x_vle, y_vle, kind=kind, bounds_error=False, fill_value=(0, 1))
    return lambda x: np.clip(f_eq(x), 0, 1)


# === MATERIAL BALANCE ===
def calculate_material_balance(feed_comp, xb, xd, feed_rate, mw1, mw2):
    """Calculate material balance for distillation column"""
    mw_feed = feed_comp * mw1 + (1 - feed_comp) * mw2
    F_mol = feed_rate * 1000 / mw_feed  # kg/hr -> mol/hr
    
    # Material balance
    try:
        D = F_mol * (feed_comp - xb) / (xd - xb)
        B = F_mol - D
    except ZeroDivisionError:
        st.error("Error in material balance: Distillate and bottoms purity are too close. (xd - xb) is zero.")
        return 0, 0, 0
    
    return D, B, F_mol


# === MINIMUM REFLUX RATIO ===
def calculate_minimum_reflux_ratio_numerical(x_vle, y_vle, feed_comp, q_value):
    """Calculate minimum reflux ratio using Underwood method"""
    if len(x_vle) < 3:
        # Fallback to simple approximation
        return 1.2
        
    # Calculate relative volatility profile
    alpha_vals = []
    x_vals = []
    
    for i in range(len(x_vle)):
        if 0.001 < x_vle[i] < 0.999 and 0.001 < y_vle[i] < 0.999:
            try:
                alpha_local = (y_vle[i] / x_vle[i]) / ((1 - y_vle[i]) / (1 - x_vle[i]))
                if alpha_local > 1.0 and alpha_local < 50:  # Reasonable bounds
                    alpha_vals.append(alpha_local)
                    x_vals.append(x_vle[i])
            except (ZeroDivisionError, ValueError):
                continue
    
    if len(alpha_vals) < 2:
        return 1.2  # Fallback
        
    # Interpolate alpha function
    alpha_func = interp1d(x_vals, alpha_vals, kind='linear', 
                          bounds_error=False, fill_value='extrapolate')
    
    # Get effective alpha at feed composition
    try:
        alpha_eff = float(alpha_func(feed_comp))
        if not np.isfinite(alpha_eff) or alpha_eff <= 1.0:
            alpha_eff = np.mean(alpha_vals)
    except:
        alpha_eff = np.mean(alpha_vals)
    
    # Underwood equation
    def underwood(theta):
        try:
            term1 = alpha_eff * feed_comp / (alpha_eff - theta)
            term2 = (1 - feed_comp) / (1 - theta)
            return term1 + term2 - (1 - q_value)
        except (ZeroDivisionError, ValueError):
            return 1e6
    
    try:
        # Find theta between 1 and alpha_eff
        theta_min = 1.001
        theta_max = alpha_eff - 0.001
        
        if theta_max <= theta_min:
            theta_max = alpha_eff * 0.95
            
        theta_sol = root_scalar(underwood, bracket=[theta_min, theta_max], 
                                 method='brentq', xtol=1e-6).root
        
        # Calculate minimum reflux
        Rmin = (alpha_eff * feed_comp) / (alpha_eff - theta_sol) + \
               (1 - feed_comp) / (1 - theta_sol) - 1
        
    except Exception as e:
        # Fallback calculation using Fenske equation approximation
        try:
            Rmin = 1.2 / (alpha_eff - 1) * np.log((feed_comp / (1 - feed_comp)) / (0.05 / 0.95))
        except ZeroDivisionError:
            Rmin = 1.2 # If alpha_eff is 1
    
    return max(Rmin, 0.1)


# === MINIMUM STAGES (FENSKE EQUATION) ===
def calculate_minimum_stages_vle(x_vle, y_vle, xd, xb):
    """Calculate minimum stages using step-by-step calculation on VLE curve"""
    if len(x_vle) < 3:
        # Fallback using average relative volatility
        alpha_avg = 2.0  # Default assumption
        try:
            return np.log((xd/(1-xd)) * ((1-xb)/xb)) / np.log(alpha_avg)
        except (ZeroDivisionError, ValueError):
            return 10
        
    # Create equilibrium function
    eq_func = interp1d(x_vle, y_vle, kind='linear', bounds_error=False, 
                       fill_value=(0, 1))
    inv_eq_func = interp1d(y_vle, x_vle, kind='linear', bounds_error=False, 
                           fill_value=(0, 1))
    
    # Step from xb to xd
    x = xb
    n = 0
    max_stages = 100
    
    while x < xd and n < max_stages:
        # Equilibrium step: x -> y
        y = float(eq_func(x))
        if not np.isfinite(y) or y <= x:
            break
            
        # Operating line step: y -> x (total reflux)
        x_new = float(inv_eq_func(y))
        if not np.isfinite(x_new) or x_new <= x:
            break
            
        x = x_new
        n += 1
        
        # Safety check
        if abs(x - xd) < 1e-6:
            break
            
    return min(n, max_stages) if n > 0 else 10  # Minimum reasonable value


# === GILLILAND CORRELATION FOR ACTUAL STAGES ===
def calculate_actual_stages(Nmin, R, Rmin):
    """Calculate actual stages using Gilliland correlation"""
    if Rmin <= 0:
        Rmin = 0.1
    
    if (R + 1) == 0: # Avoid division by zero
        return Nmin * 2

    X = (R - Rmin) / (R + 1)
    
    if X < 0 or X > 1: # X must be between 0 and 1 for correlation
        return Nmin * 2 # Fallback to a safety factor

    Y = 0.75 * (1 - X**0.5668)
    
    if Y >= 1:
        return Nmin * 2  # Safety factor
        
    if (1 - Y) == 0: # Avoid division by zero
        return Nmin * 2

    N_actual = (Nmin + Y) / (1 - Y)
    return max(N_actual, Nmin)


# === MCCABE-THIELE STAGE STEPPING ===
def calculate_stages_mccabe(x_vle, y_vle, R, xd, xb, feed_comp, q_value, F_mol, D):
    """Perform McCabe-Thiele stage stepping calculation"""
    if len(x_vle) < 3:
        return [xd, xb], [xd, xb], 10  # Fallback
        
    # Create interpolation functions
    equilibrium_func = interp1d(x_vle, y_vle, kind='linear', 
                                 fill_value=(0, 1), bounds_error=False)
    
    # Inverse equilibrium function
    x_inv = interp1d(y_vle, x_vle, kind='linear', 
                     fill_value=(0, 1), bounds_error=False)

    # Operating lines
    def rectifying_line(x): # Argument should be x, not y for operating line equation
        return (R/(R+1)) * x + xd/(R+1)

    # Stripping line parameters
    B = F_mol - D
    L_bar = R * D + q_value * F_mol
    V_bar = L_bar - B
    
    if V_bar <= 0:  # Safety check
        V_bar = 0.1 * F_mol
    
    def stripping_line(x): # Argument should be x, not y
        if V_bar == 0: return 1e6 # Prevent division by zero
        return (L_bar/V_bar) * x - (B * xb / V_bar)

    # q-line intersection (feed stage location)
    # The q-line itself is not explicitly drawn, but its intersection point
    # with the operating lines is used to switch between them.
    # The equation for the q-line is y = q/(q-1) * x - xf/(q-1) if q != 1.
    # For saturated liquid feed (q=1), the q-line is vertical at x=xf.
    
    # Find intersection point of operating lines (which happens at feed composition for ideal columns)
    # The intersection of the rectifying and stripping lines *is* the point on the q-line
    # where the operating lines meet. This point's x-coordinate is the feed composition, xf.
    # Its y-coordinate can be found from either operating line at x=xf.
    
    x_intersect = feed_comp
    y_intersect = rectifying_line(x_intersect) # Use rectifying line at feed comp to find y_intersect

    # McCabe-Thiele stepping
    x_coords = [xd] # These will store the x-coordinates of the steps
    y_coords = [xd] # These will store the y-coordinates of the steps
    
    x_current = xd
    stage_count = 0
    max_stages = 200 # Increased max stages for robustness

    # Step down the rectifying section
    while x_current > x_intersect and stage_count < max_stages:
        y_eq = float(equilibrium_func(x_current))
        if not np.isfinite(y_eq) or y_eq < x_current: # y_eq should be >= x_current
            break
        
        x_coords.append(x_current)
        y_coords.append(y_eq)
        
        x_next_op = rectifying_line(y_eq) # Vertical step from y_eq to operating line
        
        if not np.isfinite(x_next_op) or x_next_op >= x_current: # Should be decreasing
            break
        
        x_coords.append(x_next_op)
        y_coords.append(y_eq)
        x_current = x_next_op
        stage_count += 1
        
        if abs(x_current - x_intersect) < 1e-4: # If very close to feed, switch
             break

    # If we haven't reached the feed composition or have crossed it, ensure we are at the feed point
    if x_current > x_intersect:
        x_coords.append(x_intersect)
        y_coords.append(equilibrium_func(x_intersect))
        x_coords.append(x_intersect)
        y_coords.append(y_intersect)
    
    x_current = x_intersect # Start stripping section from intersection point
    
    # Step down the stripping section
    while x_current > xb and stage_count < max_stages:
        y_eq = float(equilibrium_func(x_current))
        if not np.isfinite(y_eq) or y_eq < x_current:
            break
            
        x_coords.append(x_current)
        y_coords.append(y_eq)
        
        x_next_op = stripping_line(y_eq)
        
        if not np.isfinite(x_next_op) or x_next_op >= x_current:
            break
        
        x_coords.append(x_next_op)
        y_coords.append(y_eq)
        x_current = x_next_op
        stage_count += 1
        
        if abs(x_current - xb) < 1e-6:
            break

    # Ensure the last point reaches xb
    if x_coords[-1] > xb:
        x_coords.append(xb)
        y_coords.append(equilibrium_func(xb))
        x_coords.append(xb)
        y_coords.append(xb) # y=x line for bottom

    return x_coords, y_coords, max(stage_count, 1)


# === ENERGY AND COST CALCULATIONS ===
def calculate_energy_and_cost(D, R, feed_comp, q_value, F_mol, mw1, mw2, 
                               xd, xb, n_stages, op_hours=8000):
    """Calculate energy requirements and costs"""
    # Estimate average heat of vaporization (Trouton's rule)
    # Using typical boiling points if not available
    bp_dict = {
        'ethanol': 78.4, 'water': 100.0, 'methanol': 64.7,
        'benzene': 80.1, 'toluene': 110.6, 'n-hexane': 68.7,
        'n-octane': 125.7
    }
    
    # Default boiling points - these won't be used if comp1/comp2 are in bp_dict
    Tb1_C = bp_dict.get(comp1.lower(), 78.4)
    Tb2_C = bp_dict.get(comp2.lower(), 100.0)

    Tb1 = Tb1_C + 273.15 # K
    Tb2 = Tb2_C + 273.15 # K
    
    # Estimate enthalpy of vaporization (J/mol) - using average of pure components
    dH1 = 88 * Tb1  # Trouton's rule
    dH2 = 88 * Tb2
    avg_dH_vap = feed_comp * dH1 + (1 - feed_comp) * dH2

    # Heat duties
    V = (R + 1) * D  # Vapor flow in rectifying section
    Q_cond = V * avg_dH_vap / 3.6e6  # Convert J/hr to kWh (1 J = 1 Ws, 1 kWh = 3.6e6 J)
    
    # For reboiler, consider vapor flow out of reboiler
    # L_bar is liquid flow in stripping section, V_bar is vapor flow in stripping section
    # Q_reb = V_bar * avg_dH_vap / 3.6e6
    # A simpler approximation for Q_reb often used in this context is (V_rect + F_liq)*dH_vap
    # where F_liq is the liquid portion of the feed.
    # Given q_value is 1.0 (saturated liquid feed) implies all feed is liquid.
    # So, V_bar = V + F_vapor = V (since F_vapor = 0)
    # The more accurate relation for reboiler duty from energy balance is
    # Q_reb = (L_bar + B)*h_B - L_bar*h_L_feed_temp - B*h_B_feed_temp
    # Using the vapor flow leaving the reboiler (V_bar) and latent heat:
    Q_reb = V * avg_dH_vap / 3.6e6 # For saturated liquid feed, V_bar = V

    # Equipment costs (rough estimates) - simplified for conceptual design
    tower_cost = 15000 * (n_stages ** 0.8) if n_stages > 0 else 15000
    condenser_cost = 5000 * (Q_cond ** 0.65) if Q_cond > 0 else 5000
    reboiler_cost = 6000 * (Q_reb ** 0.7) if Q_reb > 0 else 6000
    total_equip_cost = tower_cost + condenser_cost + reboiler_cost

    # Operating costs
    energy_cost_per_kwh = 0.10  # $/kWh
    energy_cost_hr = (Q_cond + Q_reb) * energy_cost_per_kwh
    annual_energy_cost = energy_cost_hr * op_hours
    annual_operating_cost = annual_energy_cost * 1.2  # Including maintenance, labor, etc.

    # Product costs
    avg_mw_distillate = xd * mw1 + (1 - xd) * mw2
    distillate_kg_hr = D * avg_mw_distillate / 1000
    cost_per_kg = energy_cost_hr / distillate_kg_hr if distillate_kg_hr > 0 else 0
    payback_years = total_equip_cost / annual_operating_cost if annual_operating_cost > 0 else float('inf')

    return {
        'Q_cond': Q_cond,
        'Q_reb': Q_reb,
        'total_equip_cost': total_equip_cost,
        'energy_cost_hr': energy_cost_hr,
        'annual_energy_cost': annual_energy_cost,
        'annual_operating_cost': annual_operating_cost,
        'cost_per_kg': cost_per_kg,
        'payback_years': payback_years,
        'distillate_kg_hr': distillate_kg_hr,
        'V': V
    }


# === PLOTTING FUNCTION ===
def plot_mccabe_thiele_nist(x_vle, y_vle, x_stages, y_stages, R, xd, xb, feed_comp):
    """Plot McCabe-Thiele diagram"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # VLE curve
    ax.plot(x_vle, y_vle, 'b-', linewidth=2, label='VLE Curve')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='y = x')
    
    # Operating lines
    # Rectifying Line: y = (R/(R+1))x + xd/(R+1)
    x_rect = np.linspace(feed_comp, 1, 100)
    y_rect = (R/(R+1)) * x_rect + xd/(R+1)
    ax.plot(x_rect, y_rect, 'g-', linewidth=1.5, label=f'Rectifying Line (R={R:.2f})')

    # Stripping Line: y = (L_bar/V_bar)x - (B*xb)/V_bar
    # To draw the stripping line correctly, we need the intersection point or xb and feed_comp.
    # The stripping line connects (xb, xb) to the intersection point of rectifying line and q-line.
    
    # Calculate intersection point (feed point on operating lines)
    # For saturated liquid feed (q=1), the feed line is vertical at x=feed_comp.
    # The intersection point is (feed_comp, y_rect_at_feed_comp)
    y_feed_intersect = (R/(R+1)) * feed_comp + xd/(R+1)

    # Calculate stripping line parameters again to ensure consistency if not calculated before.
    # This requires D, B, F_mol, q_value. For this plotting function,
    # let's assume a simplified approach or pass them if available.
    # For simplicity, we'll draw it from xb to the feed point on the operating line.
    
    # To draw the stripping line, we need to find its y-intercept or use two points.
    # The two points are (xb, xb) and (feed_comp, y_feed_intersect).
    x_strip = np.linspace(0, feed_comp, 100) # x-range for stripping line
    
    # Recalculate L_bar and V_bar for stripping line explicitly here for clarity
    D_dummy, B_dummy, F_mol_dummy = calculate_material_balance(feed_comp, xb, xd, 1000, mw_dict['ethanol'], mw_dict['water']) # Use dummy values to get D
    # For a saturated liquid feed (q=1), L = R*D and V = (R+1)*D (rectifying section)
    # L_bar = L + F = R*D + F
    # V_bar = V = (R+1)*D
    # The original stripping_line function already calculates L_bar/V_bar correctly based on F_mol, D.
    
    # Find the slope (L_bar/V_bar) and y-intercept (-B*xb/V_bar)
    # We need D and F_mol from the main function. These are not directly passed to plot_mccabe_thiele_nist.
    # Let's compute them or pass them down. For a standalone plot function, it's better to pass.
    # For now, let's use a simpler approach that might not be perfectly rigorous without those values:
    # Use the two points (xb, xb) and (feed_comp, y_feed_intersect) to define the stripping line.
    
    # Slope of stripping line: (y_feed_intersect - xb) / (feed_comp - xb)
    # If feed_comp == xb, this will fail. Handle this case.
    if feed_comp - xb != 0:
        m_strip = (y_feed_intersect - xb) / (feed_comp - xb)
        c_strip = xb - m_strip * xb
        y_strip = m_strip * x_strip + c_strip
        ax.plot(x_strip, y_strip, 'c-', linewidth=1.5, label='Stripping Line')
    else:
        st.warning("Cannot draw stripping line: feed composition and bottoms composition are the same.")

    # Stage steps
    # The x_stages and y_stages from calculate_stages_mccabe already contain the pairs
    # (x_eq, y_eq) and (x_op, y_op) that form the steps.
    
    for i in range(0, len(x_stages) - 1, 2):
        # Horizontal step: from (x_op, y_op) to (x_eq, y_eq) - (x_op, y_op) is (x_stages[i], y_stages[i])
        # Equilibrium step moves horizontally from operating line to VLE curve
        ax.hlines(y_stages[i], x_stages[i+1], x_stages[i], colors='red', linestyles='-', alpha=0.7, linewidth=1)
        
        # Vertical step: from (x_eq, y_eq) to (x_next_op, y_eq)
        # Operating line step moves vertically from VLE curve back to operating line
        if i+2 < len(x_stages):
             ax.vlines(x_stages[i+1], y_stages[i+2], y_stages[i+1], colors='red', linestyles='-', alpha=0.7, linewidth=1)
        
    # Feed line and specifications
    ax.axvline(x=feed_comp, color='orange', linestyle=':', alpha=0.8, label=f'Feed (x={feed_comp:.3f})')
    ax.axvline(x=xd, color='purple', linestyle=':', alpha=0.8, label=f'Distillate (x={xd:.3f})')
    ax.axvline(x=xb, color='brown', linestyle=':', alpha=0.8, label=f'Bottoms (x={xb:.3f})')
    
    # Add stages as discrete points
    for i in range(0, len(x_stages), 2): # Plot points on the VLE curve
        ax.plot(x_stages[i], y_stages[i], 'ro', markersize=4, alpha=0.7)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Liquid mole fraction, x', fontsize=12)
    ax.set_ylabel('Vapor mole fraction, y', fontsize=12)
    ax.set_title('McCabe-Thiele Diagram', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig) # Display the plot in Streamlit


# === Streamlit App Layout ===
st.set_page_config(layout="wide", page_title="Distillation Column Design")

st.title("🧪 Distillation Column Design Tool")
st.markdown("---")

st.sidebar.header("Mixture Selection")
# Available mixtures
valid_pairs = [
    ('ethanol', 'water'),
    ('methanol', 'water'),
    ('benzene', 'toluene'),
    ('n-hexane', 'n-octane')
]

mixture_options = [f"{c1.title()} / {c2.title()}" for c1, c2 in valid_pairs]
selected_mixture_index = st.sidebar.selectbox("Choose a mixture:", range(len(mixture_options)), format_func=lambda x: mixture_options[x])

comp1, comp2 = valid_pairs[selected_mixture_index]

# Molecular weights (using a dictionary as fallback if chemicals library is not available)
mw_dict = {
    'ethanol': 46.07, 'water': 18.02, 'methanol': 32.04,
    'benzene': 78.11, 'toluene': 92.14, 'n-hexane': 86.18,
    'n-octane': 114.23
}

if CHEMICALS_AVAILABLE:
    try:
        mw1 = MolarMass(comp1)
        mw2 = MolarMass(comp2)
    except:
        mw1 = mw_dict.get(comp1, 46.07)
        mw2 = mw_dict.get(comp2, 18.02)
else:
    mw1 = mw_dict.get(comp1, 46.07)
    mw2 = mw_dict.get(comp2, 18.02)

st.sidebar.header("Design Parameters")
feed_comp = st.sidebar.slider(f"Feed mole fraction of {comp1.title()}", 0.01, 0.99, 0.4, 0.01)
feed_rate = st.sidebar.number_input("Feed rate (kg/hr)", min_value=1.0, value=1000.0)
xd = st.sidebar.slider(f"Distillate purity (mole fraction {comp1.title()})", feed_comp + 0.01, 0.999, 0.85, 0.001)
xb = st.sidebar.slider(f"Bottoms impurity (mole fraction {comp1.title()})", 0.001, feed_comp - 0.01, 0.05, 0.001)
rr_mult = st.sidebar.slider("Reflux ratio multiplier (R/Rmin)", 1.05, 5.0, 1.5, 0.05)
P_atm = st.sidebar.number_input("Pressure (atm)", min_value=0.1, value=1.0)

P = P_atm * 101325  # Convert to Pa

st.markdown(f"## Design for {comp1.title()} / {comp2.title()} System")

if st.button("Run Design Calculation"):
    st.markdown("### Calculation Results")
    st.markdown("---")
    
    # Get VLE Data
    with st.spinner(f"Fetching VLE data for {comp1.title()} + {comp2.title()} at {P_atm} atm..."):
        x_vle, y_vle, T_vle = get_vle_data_nist(comp1, comp2, P=P)

    # Fallback to ideal model if NIST data unavailable
    if x_vle is None or len(x_vle) < 3:
        st.warning("❌ Using ideal VLE model (constant relative volatility) due to unavailable or insufficient NIST data.")
        
        bp1 = mw_dict.get(comp1.lower(), 78.4) # Using boiling points from mw_dict
        bp2 = mw_dict.get(comp2.lower(), 100.0)
        alpha = calculate_relative_volatility_ideal(bp1, bp2)
        
        x_vle = np.linspace(0, 1, 21)
        y_vle = alpha * x_vle / (1 + (alpha - 1) * x_vle)
        
        st.info(f"Using relative volatility α = {alpha:.2f}")
    else:
        st.success("✅ Successfully loaded VLE data.")

    # Perform calculations
    st.markdown("### Distillation Parameters")
    try:
        # Material balance
        D, B, F_mol = calculate_material_balance(feed_comp, xb, xd, feed_rate, mw1, mw2)
        
        # Minimum reflux ratio
        Rmin = calculate_minimum_reflux_ratio_numerical(x_vle, y_vle, feed_comp, q_value=1.0) # q_value=1.0 for saturated liquid feed
        R = rr_mult * Rmin
        
        # Minimum stages
        Nmin = calculate_minimum_stages_vle(x_vle, y_vle, xd, xb)
        
        # Actual stages (Gilliland)
        N_actual = calculate_actual_stages(Nmin, R, Rmin)
        
        # McCabe-Thiele stepping
        x_stages, y_stages, n_stages = calculate_stages_mccabe(
            x_vle, y_vle, R, xd, xb, feed_comp, 1.0, F_mol, D) # q_value=1.0 for saturated liquid feed
        
        # Economics
        econ = calculate_energy_and_cost(
            D, R, feed_comp, 1.0, F_mol, mw1, mw2, xd, xb, n_stages, op_hours=8000)

        # Display results in Streamlit columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Feed and Product Streams")
            st.write(f"**Feed Rate:** {feed_rate:.0f} kg/hr ({F_mol:.0f} mol/hr)")
            st.write(f"**Feed Composition:** {feed_comp:.3f} mole fraction {comp1.title()}")
            st.write(f"**Distillate Purity:** {xd:.3f} mole fraction {comp1.title()}")
            st.write(f"**Bottoms Composition:** {xb:.3f} mole fraction {comp1.title()}")
            st.write(f"**Distillate Rate:** {econ['distillate_kg_hr']:.1f} kg/hr ({D:.0f} mol/hr)")
            st.write(f"**Bottoms Rate:** {(feed_rate - econ['distillate_kg_hr']):.1f} kg/hr ({B:.0f} mol/hr)")
        
        with col2:
            st.subheader("Column Design Summary")
            st.write(f"**Minimum Reflux Ratio:** {Rmin:.2f}")
            st.write(f"**Operating Reflux Ratio:** {R:.2f}")
            st.write(f"**Minimum Stages (Fenske):** {Nmin:.1f}")
            st.write(f"**Actual Stages (Gilliland):** {N_actual:.1f}")
            st.write(f"**Theoretical Stages (McCabe-Thiele):** {n_stages}")
            st.write(f"**Condenser Duty:** {econ['Q_cond']:.1f} kW")
            st.write(f"**Reboiler Duty:** {econ['Q_reb']:.1f} kW")
            st.write(f"**Vapor Flow Rate (Rectifying):** {econ['V']:.0f} mol/hr")

        st.markdown("### Economic Analysis")
        col3, col4 = st.columns(2)
        with col3:
            st.write(f"**Total Equipment Cost:** ${econ['total_equip_cost']:,.0f}")
            st.write(f"**Energy Cost (per hour):** ${econ['energy_cost_hr']:.2f}/hr")
            st.write(f"**Annual Energy Cost:** ${econ['annual_energy_cost']:,.0f}")
        with col4:
            st.write(f"**Annual Operating Cost (Est.):** ${econ['annual_operating_cost']:,.0f}")
            st.write(f"**Cost per kg Product:** ${econ['cost_per_kg']:.4f}")
            st.write(f"**Estimated Payback Period:** {econ['payback_years']:.1f} years")

        st.markdown("---")
        st.subheader("McCabe-Thiele Diagram")
        plot_mccabe_thiele_nist(x_vle, y_vle, x_stages, y_stages, R, xd, xb, feed_comp)

        # Export to Excel
        results_df = pd.DataFrame({
            'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            'System': [f"{comp1.title()}/{comp2.title()}"],
            'Pressure_atm': [P_atm],
            'Feed_rate_kg_hr': [feed_rate],
            'Feed_composition': [feed_comp],
            'Distillate_purity': [xd],
            'Bottoms_composition': [xb],
            'Min_reflux_ratio': [Rmin],
            'Operating_reflux_ratio': [R],
            'Min_stages_Fenske': [Nmin],
            'Actual_stages_Gilliland': [N_actual],
            'Theoretical_stages_McCabe': [n_stages],
            'Distillate_rate_kg_hr': [econ['distillate_kg_hr']],
            'Bottoms_rate_kg_hr': [(feed_rate - econ['distillate_kg_hr'])],
            'Condenser_duty_kW': [econ['Q_cond']],
            'Reboiler_duty_kW': [econ['Q_reb']],
            'Total_Equipment_cost_USD': [econ['total_equip_cost']],
            'Energy_cost_per_hr_USD': [econ['energy_cost_hr']],
            'Annual_Energy_Cost_USD': [econ['annual_energy_cost']],
            'Annual_Operating_Cost_USD': [econ['annual_operating_cost']],
            'Cost_per_kg_product_USD': [econ['cost_per_kg']],
            'Payback_years': [econ['payback_years']]
        })
        
        excel_filename = f"distillation_results_{comp1}_{comp2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        # Convert DataFrame to Excel in-memory to allow direct download
        output = results_df.to_excel(excel_filename, index=False, engine='xlsxwriter')
        
        # Streamlit download button
        # The above 'output' is not directly a file. We need to use BytesIO for in-memory handling.
        from io import BytesIO
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            results_df.to_excel(writer, index=False, sheet_name='Distillation Results')
        buffer.seek(0)
        
        st.download_button(
            label="Download Results as Excel",
            data=buffer,
            file_name=excel_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"❌ An error occurred during calculations: {e}")
        st.info("Please check your input parameters and try again. Ensure Distillate purity > Feed composition > Bottoms impurity.")

st.markdown("---")
st.info("💡 This tool provides a conceptual design. For detailed engineering, consider rigorous simulation software.")
