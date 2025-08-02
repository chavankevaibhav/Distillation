# Save the distillation functions to a Python file
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# Constants
R_gas = 8.314  # J/mol·K

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

def equilibrium(x, alpha):
    return alpha * x / (1 + (alpha - 1) * x)

def rectifying_line(x, R, xd):
    return (R/(R+1)) * x + xd/(R+1)

def stripping_line(x, L_bar, V_bar, B, xb):
    return (L_bar / V_bar) * x - (B * xb) / V_bar

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

def calculate_energy_and_cost(D, R, feed_comp, q_value, F_mol, mw1, mw2, xd, xb, n_stages, bp1, bp2):
    avg_dH_vap = feed_comp * 88 * (bp1 + 273.15) + (1 - feed_comp) * 88 * (bp2 + 273.15)  # J/mol
    V = (R + 1) * D  # Vapor flow (mol/hr)
    Q_cond = V * avg_dH_vap / 3.6e6  # kWh
    Q_reb = (V + (1 - q_value) * F_mol) * avg_dH_vap / 3.6e6  # kWh

    tower_cost = 15000 * (n_stages ** 0.8)
    condenser_cost = 5000 * (Q_cond ** 0.65)
    reboiler_cost = 6000 * (Q_reb ** 0.7)
    total_equip_cost = tower_cost + condenser_cost + reboiler_cost
    energy_cost_hr = (Q_cond + Q_reb) * 0.10  # $0.10/kWh

    distillate_kg = D * (xd * mw1 + (1 - xd) * mw2) / 1000  # kg/hr
    cost_per_kg = energy_cost_hr / distillate_kg if distillate_kg > 0 else 0

    return Q_cond, Q_reb, total_equip_cost, energy_cost_hr, cost_per_kg

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

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x, y_eq, 'b', lw=2, label='Equilibrium')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.7)

    ax.plot(x, y_rect, 'g', lw=2, label='Rectifying')
    ax.plot(x, y_strip, 'r', lw=2, label='Stripping')

    if abs(q - 1.0) < 1e-6:
        ax.axvline(x=zf, color='m', linestyle='--', lw=2, label='q-line')
    elif abs(q - 1.0) > 1e-3:
        ax.plot(qx, qy, 'm--', lw=2, label='q-line')

    ax.plot(stairs_x, stairs_y, 'o-', color='orange', markersize=4,
            label='Stages')

    if feed_stage is not None:
        idx = 2 * feed_stage + 1
        ax.scatter(stairs_x[idx], stairs_y[idx], color='purple',
                   s=80, zorder=5, label=f'Feed stage ({feed_stage})')

    for i in range(1, n_stages + 1):
        idx = 4 * i - 2
        if idx < len(stairs_x):
            ax.text(stairs_x[idx] + 0.01, stairs_y[idx], str(i),
                    fontsize=9, color='navy')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Liquid mole fraction (light key)')
    ax.set_ylabel('Vapor mole fraction (light key)')
    ax.set_title('McCabe-Thiele Diagram')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, n_stages, feed_stage

def calculate_feed_stage(alpha, R, xd, xb, zf, q, F_mol, D, B):
    """
    Calculates the estimated feed stage based on the intersection of the
    q-line and operating lines.
    """
    # Calculate intersection of q-line and rectifying line
    if abs(q - 1.0) < 1e-6: # q = 1 (saturated liquid), q-line is vertical at x = zf
        x_feed = zf
    elif abs(q + R) < 1e-6: # Avoid division by zero if q+R is close to zero
         x_feed = (xd*q - xd + zf*R + zf) / 1e-6 # Use a small number instead of 0
    else:
        x_feed = (xd*q - xd + zf*R + zf) / (q + R)

    y_feed = _rectifying(x_feed, R, xd) # Use the rectifying line to find y_feed

    # Calculate stages to find the one closest to the intersection point
    x_stages, y_stages, n_stages = calculate_stages(alpha, R, xd, xb, zf, q, F_mol, D, B)

    feed_stage = None
    for i in range(1, len(x_stages) - 1, 2):
        if x_stages[i] >= x_feed and x_stages[i+2] <= x_feed:
            feed_stage = (i + 1) // 2
            break
        elif x_stages[i+2] > x_feed and x_stages[i] > x_feed and i==1: # if the intersection is above the first stage
            feed_stage = 0
            break

    return feed_stage
