# app.py - Streamlit App for McCabe-Thiele Distillation Column Design

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import requests
from bs4 import BeautifulSoup

# Constants
R_gas = 8.314  # J/mol·K

# -----------------------------
# Utility Functions (from your original code)
# -----------------------------

def calculate_material_balance(feed_comp, xb, xd, feed_rate, mw1, mw2):
    mw_feed = feed_comp * mw1 + (1 - feed_comp) * mw2
    F_mol = feed_rate * 1000 / mw_feed
    if abs(xd - xb) < 1e-9:
        D = 0.0
        B = F_mol
    else:
        D = F_mol * (feed_comp - xb) / (xd - xb)
        B = F_mol - D
    D = max(0.0, D)
    B = max(0.0, B)
    if D == 0.0:
        B = F_mol
    return D, B, F_mol

def calculate_minimum_reflux_ratio(alpha, zf, q):
    if abs(q - 1.0) < 1e-9:
        x_int = zf
        y_int = alpha * x_int / (1 + (alpha - 1) * x_int)
        if abs(y_int - x_int) < 1e-9:
            return float('inf')
        Rmin = (xd - y_int) / (y_int - x_int)
    else:
        if abs(alpha - 1.0) < 1e-9:
            return float('inf')
        A = (q / (q - 1.0)) * (alpha - 1.0)
        B = (q / (q - 1.0)) - (zf / (q - 1.0)) * (alpha - 1.0) - alpha
        C = - (zf / (q - 1.0))
        disc = B**2 - 4*A*C
        if disc < 0:
            return float('inf')
        sqrt_disc = np.sqrt(disc)
        x1 = (-B + sqrt_disc) / (2 * A)
        x2 = (-B - sqrt_disc) / (2 * A)
        valid_x = [x for x in [x1, x2] if xb <= x <= xd + 1e-9]
        if not valid_x:
            return float('inf')
        x_int = min(valid_x)
        y_int = alpha * x_int / (1 + (alpha - 1) * x_int)
        if abs(y_int - x_int) < 1e-9:
            return float('inf')
        Rmin = (xd - y_int) / (y_int - x_int)
    return max(Rmin, 1e-9)

def calculate_actual_reflux_ratio(Rmin, rr_mult):
    return Rmin * rr_mult if Rmin != float('inf') else float('inf')

def calculate_minimum_stages(xd, xb, alpha):
    if xb <= 0 or xd >= 1 or alpha <= 1:
        return float('inf')
    try:
        return np.log((xd / (1 - xd)) * ((1 - xb) / xb)) / np.log(alpha)
    except:
        return float('inf')

def calculate_actual_stages(Nmin, R, Rmin):
    if R <= Rmin or Nmin == float('inf') or Rmin == float('inf'):
        return float('inf')
    X = (R - Rmin) / (R + 1)
    X = max(0.0, min(X, 0.95))
    Y = 0.5458 - 0.1173*X + 0.0324*X**2 + 0.0015*X**3 - 0.0019*X**4
    Y = min(Y, 0.99)
    if abs(1 - Y) < 1e-9:
        return float('inf')
    N = (Nmin + Y) / (1 - Y)
    return max(Nmin, np.ceil(N))

def _rectifying(x, R, xd):
    return (R / (R + 1)) * x + (xd / (R + 1))

def _stripping(x, L_bar, V_bar, xb):
    if abs(V_bar) < 1e-9:
        return np.nan
    return (L_bar / V_bar) * x - (xb * (L_bar - V_bar)) / V_bar  # B = L_bar - V_bar

def plot_mccabe_thiele(alpha, xd, xb, R, zf, q, D, B, F_mol):
    x = np.linspace(0, 1, 500)
    y_eq = alpha * x / (1 + (alpha - 1) * x)

    L_bar = R * D + q * F_mol
    V_bar = L_bar - B

    fig, ax = plt.subplots(figsize=(8, 8))

    # Equilibrium curve
    ax.plot(x, y_eq, label="Equilibrium Curve", color="blue")

    # Diagonal
    ax.plot([0, 1], [0, 1], 'k--', label="y = x")

    # Rectifying line
    y_rect = _rectifying(x, R, xd)
    ax.plot(x, y_rect, label=f"Rectifying Line (R={R:.2f})", color="green")

    # Stripping line
    y_strip = _stripping(x, L_bar, V_bar, xb)
    ax.plot(x, y_strip, label="Stripping Line", color="orange")

    # q-line
    if abs(q - 1.0) < 1e-9:
        ax.axvline(x=zf, color="purple", linestyle="-.", label="q-line (saturated liquid)")
    else:
        y_q = (q / (q - 1)) * x - (zf / (q - 1))
        ax.plot(x, y_q, color="purple", linestyle="-.", label=f"q-line (q={q:.2f})")

    # Stage stepping
    n_stages = 0
    feed_stage = None
    x_step = xd
    y_step = xd
    ax.plot(x_step, y_step, 'ro')

    while x_step > xb:
        # Horizontal to equilibrium
        y_step = _rectifying(x_step, R, xd)
        if y_step > 1 or y_step < 0:
            break
        ax.plot([x_step, x_step], [x_step, y_step], 'r-', lw=1)
        # Vertical to operating line
        x_next = y_step * (1 + (alpha - 1) * x_step) / alpha
        ax.plot([x_step, x_next], [y_step, y_step], 'r-', lw=1)
        n_stages += 1
        if feed_stage is None and x_next <= zf:
            feed_stage = n_stages
        x_step = x_next

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Liquid mole fraction (x)")
    ax.set_ylabel("Vapor mole fraction (y)")
    ax.set_title("McCabe-Thiele Diagram")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    return n_stages, feed_stage

def fetch_nist_data(component_name):
    base_url = "https://webbook.nist.gov/cgi/cbook.cgi?Name="
    search_url = f"{base_url}{requests.utils.quote(component_name)}"

    try:
        response = requests.get(search_url, timeout=5)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        boiling_point = None
        molecular_weight = None

        # Extract molecular weight
        mw_tag = soup.find(string="Molecular formula")
        if mw_tag:
            parent = mw_tag.find_parent('tr').find_next_sibling('tr')
            if parent:
                text = parent.get_text()
                import re
                m = re.search(r"(\d+\.\d+)", text)
                if m:
                    molecular_weight = float(m.group(1))

        # Extract boiling point
        for h in soup.find_all(['h2', 'h3'], string=lambda t: t and "Phase change data" in t):
            table = h.find_next('table')
            if table:
                rows = table.find_all('tr')
                for row in rows:
                    th = row.find('th')
                    if th and "Boiling Point" in th.get_text():
                        td = row.find('td')
                        if td:
                            txt = td.get_text()
                            m = re.search(r"(\d+\.?\d*)\s*C", txt)
                            if m:
                                boiling_point = float(m.group(1))
                                break
        if boiling_point and molecular_weight:
            return {"bp": boiling_point, "mw": molecular_weight}
        else:
            st.warning(f"⚠️ Incomplete data for '{component_name}' from NIST.")
            return None
    except Exception as e:
        st.error(f"❌ Error fetching data for '{component_name}': {e}")
        return None

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Distillation Column Calculator", layout="wide")
st.title("⚗️ McCabe-Thiele Distillation Column Designer")

st.markdown("""
This app calculates distillation parameters using McCabe-Thiele method.
Enter component names to auto-fetch boiling points and molecular weights from [NIST WebBook](https://webbook.nist.gov).
""")

col1, col2 = st.columns(2)

with col1:
    st.header("Component 1 (Light Key)")
    comp1_name = st.text_input("Name (e.g., ethanol)", value="ethanol")
    bp1_input = st.number_input("Boiling Point (°C)", value=78.37)
    mw1_input = st.number_input("Molecular Weight (g/mol)", value=46.07)

with col2:
    st.header("Component 2 (Heavy Key)")
    comp2_name = st.text_input("Name (e.g., water)", value="water")
    bp2_input = st.number_input("Boiling Point (°C)", value=100.0)
    mw2_input = st.number_input("Molecular Weight (g/mol)", value=18.02)

if st.button("Fetch Data from NIST"):
    with st.spinner("Fetching data from NIST WebBook..."):
        data1 = fetch_nist_data(comp1_name)
        data2 = fetch_nist_data(comp2_name)
        if data1:
            bp1_input, mw1_input = data1['bp'], data1['mw']
            st.success(f"✅ {comp1_name.title()}: BP = {bp1_input:.2f}°C, MW = {mw1_input:.2f} g/mol")
        if data2:
            bp2_input, mw2_input = data2['bp'], data2['mw']
            st.success(f"✅ {comp2_name.title()}: BP = {bp2_input:.2f}°C, MW = {mw2_input:.2f} g/mol")

st.header("Process Parameters")
zf = st.slider("Feed Mole Fraction (Light Key)", 0.0, 1.0, 0.5)
xd = st.slider("Distillate Mole Fraction (Light Key)", 0.7, 1.0, 0.95)
xb = st.slider("Bottoms Mole Fraction (Light Key)", 0.0, 0.3, 0.05)
feed_rate = st.number_input("Feed Rate (kg/hr)", min_value=1.0, value=1000.0)
q = st.slider("Feed Thermal Condition (q)", 0.0, 2.0, 1.0, help="q=1: saturated liquid, q=0: saturated vapor")
rr_mult = st.slider("Reflux Ratio Multiplier (R/Rmin)", 1.1, 5.0, 1.5)

# Calculate alpha using boiling points
Tb1 = bp1_input + 273.15
Tb2 = bp2_input + 273.15
alpha = np.exp((88 * (Tb1 - Tb2)) / (R_gas * (Tb1 + Tb2) / 2))  # Simplified

# Material balance
D, B, F_mol = calculate_material_balance(zf, xb, xd, feed_rate, mw1_input, mw2_input)

# Rmin and R
Rmin = calculate_minimum_reflux_ratio(alpha, zf, q)
R = calculate_actual_reflux_ratio(Rmin, rr_mult)

# Stages
Nmin = calculate_minimum_stages(xd, xb, alpha)
N_actual = calculate_actual_stages(Nmin, R, Rmin)

# Display Results
st.header("📊 Results")
res_col1, res_col2 = st.columns(2)

with res_col1:
    st.subheader("Flow Rates")
    st.write(f"**Distillate (D):** {D:.2f} mol/hr")
    st.write(f"**Bottoms (B):** {B:.2f} mol/hr")
    st.write(f"**Feed (F):** {F_mol:.2f} mol/hr")

with res_col2:
    st.subheader("Design Parameters")
    st.write(f"**Relative Volatility (α):** {alpha:.2f}")
    st.write(f"**Min Reflux (Rmin):** {Rmin:.2f}")
    st.write(f"**Actual Reflux (R):** {R:.2f}")
    st.write(f"**Min Stages (Nmin):** {Nmin:.2f}")
    st.write(f"**Actual Stages (N):** {int(N_actual) if N_actual != float('inf') else '∞'}")

# McCabe-Thiele Plot
st.header("📈 McCabe-Thiele Diagram")
if Rmin != float('inf') and N_actual != float('inf'):
    total_stages, feed_stage = plot_mccabe_thiele(alpha, xd, xb, R, zf, q, D, B, F_mol)
    st.write(f"Estimated feed stage (from top): **{feed_stage or 'Not detected'}**")
else:
    st.warning("Cannot plot diagram due to infinite reflux or stages.")

# Energy & Cost (optional extension)
st.info("💡 Tip: Extend this app with energy cost, tower height, or rigorous VLE using Antoine equation.")
