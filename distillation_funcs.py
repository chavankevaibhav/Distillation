%%writefile distillation_funcs.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, minimize
import requests
from bs4 import BeautifulSoup

# Constants
R_gas = 8.314  # J/mol·K

def antoine_vapor_pressure(T_celsius, A, B, C):
    """
    Calculates vapor pressure in kPa using the Antoine equation.
    T_celsius: Temperature in °C
    A, B, C: Antoine coefficients
    Returns: Vapor pressure in kPa
    """
    # Antoine equation: log10(P_vp) = A - (B / (T + C))
    # T in °C, P_vp in kPa
    T_kelvin = T_celsius + 273.15 # Convert to Kelvin if coefficients are for Kelvin
    # Check coefficient basis - assume for Celsius based on typical usage, adjust if needed
    try:
        log10_Pvp_kpa = A - (B / (T_celsius + C))
        P_vp_kpa = 10**log10_Pvp_kpa
        return P_vp_kpa
    except Exception as e:
        print(f"Error calculating Antoine vapor pressure: {e}")
        return np.nan

def kpa_to_atm(p_kpa):
    """Converts pressure from kPa to atm."""
    return p_kpa / 101.325

def calculate_vle(x1, T_celsius, P_atm, antoine_coeffs1, antoine_coeffs2, gamma1=1.0, gamma2=1.0):
    """
    Calculates the equilibrium vapor mole fraction (y1) for component 1.
    x1: Liquid mole fraction of component 1
    T_celsius: Temperature in °C
    P_atm: Total pressure in atm
    antoine_coeffs1: (A, B, C) for component 1
    antoine_coeffs2: (A, B, C) for component 2
    gamma1, gamma2: Activity coefficients (default to 1 for ideal solution)
    Returns: Equilibrium vapor mole fraction y1, or None if calculation fails.
    """
    try:
        P_vp1_kpa = antoine_vapor_pressure(T_celsius, *antoine_coeffs1)
        P_vp2_kpa = antoine_vapor_pressure(T_celsius, *antoine_coeffs2)

        P_vp1_atm = kpa_to_atm(P_vp1_kpa)
        P_vp2_atm = kpa_to_atm(P_vp2_kpa)

        # Raoult's Law with activity coefficients: P * y1 = x1 * gamma1 * P_vp1
        # P * y2 = x2 * gamma2 * P_vp2
        # y1 + y2 = 1 => y2 = 1 - y1
        # P * (1 - y1) = (1 - x1) * gamma2 * P_vp2
        # P = x1 * gamma1 * P_vp1 + (1 - x1) * gamma2 * P_vp2  (Bubble Point Pressure)

        # Assuming Bubble Point calculation to find T for a given x1 and P
        # Or Rachford-Rice for flash calculation

        # For VLE calculation (given x1, T, P), we need to check if T is the bubble point temperature at P
        # Assuming ideal solution (gamma1=gamma2=1) and Raoult's Law for now:
        # P * y1 = x1 * P_vp1_atm
        # P * y2 = x2 * P_vp2_atm
        # y1 = (x1 * P_vp1_atm) / P_atm

        # More rigorously, solve for y1 using the bubble point pressure equation
        # P = x1 * gamma1 * P_vp1_atm + (1 - x1) * gamma2 * P_vp2_atm
        # If the given P_atm is not the bubble point pressure at T and x1,
        # this T is not the equilibrium temperature for this x1 at this P.

        # Let's implement the bubble point temperature calculation first, as it's needed for the stages.
        # Given x1 and P, find T such that P = x1 * gamma1 * P_vp1(T) + (1-x1) * gamma2 * P_vp2(T)
        # Then calculate y1 at that T.

        # For simplicity in this initial step, let's assume we are at the bubble point and
        # calculate y1 using the modified Raoult's Law: y1 = (x1 * gamma1 * P_vp1_atm) / P_atm
        # This assumes the provided T is the equilibrium temperature for the given x1 and P.
        # A more robust implementation would involve a bubble point calculation.

        # Let's return y1 based on Raoult's Law for now, assuming the input T is the equilibrium T.
        # This is consistent with replacing the old equilibrium function.
        y1 = (x1 * gamma1 * P_vp1_atm) / P_atm
        return y1

    except Exception as e:
        print(f"Error in calculate_vle: {e}")
        return np.nan


def nrtl_activity_coefficient(x1, T_celsius, params):
    """
    Placeholder for NRTL activity coefficient calculation.
    Currently returns ideal activity coefficients (1.0).
    params: Dictionary or tuple of NRTL parameters
    """
    # This function would implement the NRTL model
    # For now, return ideal
    return 1.0

def uniquac_activity_coefficient(x1, T_celsius, params):
    """
    Placeholder for UNIQUAC activity coefficient calculation.
    Currently returns ideal activity coefficients (1.0).
    params: Dictionary or tuple of UNIQUAC parameters
    """
    # This function would implement the UNIQUAC model
    # For now, return ideal
    return 1.0


# Update the equilibrium function to use the new VLE calculation
def equilibrium(x, T_celsius, P_atm, antoine_coeffs1, antoine_coeffs2, gamma1=1.0, gamma2=1.0):
     """
     Calculates the equilibrium vapor mole fraction (y) using calculate_vle.
     x: Liquid mole fraction of light key
     T_celsius: Temperature in °C
     P_atm: Total pressure in atm
     antoine_coeffs1: (A, B, C) for light key
     antoine_coeffs2: (A, B, C) for heavy key
     gamma1, gamma2: Activity coefficients (default to 1 for ideal solution)
     Returns: Equilibrium vapor mole fraction y.
     """
     # In a real stage calculation, T would be the bubble point temperature at liquid composition x and pressure P.
     # For the McCabe-Thiele plot and stage stepping, we need to find the equilibrium T for each x on the equilibrium curve.
     # This requires solving P_total = x1*gamma1*Pvp1(T) + x2*gamma2*Pvp2(T) for T.
     # Let's create a helper function for bubble point temperature.

     def bubble_point_temperature(x1, P_atm, antoine_coeffs1, antoine_coeffs2, gamma1_func=lambda x, T: 1.0, gamma2_func=lambda x, T: 1.0):
         """
         Calculates the bubble point temperature (°C) for a given liquid composition and pressure.
         x1: Liquid mole fraction of component 1
         P_atm: Total pressure in atm
         antoine_coeffs1: (A, B, C) for component 1
         antoine_coeffs2: (A, B, C) for component 2
         gamma1_func: Function to calculate activity coefficient for component 1 (takes x1, T_celsius)
         gamma2_func: Function to calculate activity coefficient for component 2 (takes x1, T_celsius)
         Returns: Bubble point temperature in °C.
         """
         def pressure_difference(T):
             try:
                 gamma1 = gamma1_func(x1, T)
                 gamma2 = gamma2_func(1 - x1, T) # Assuming activity coef depends on liquid comp of that component

                 P_vp1_kpa = antoine_vapor_pressure(T, *antoine_coeffs1)
                 P_vp2_kpa = antoine_vapor_pressure(T, *antoine_coeffs2)
                 P_vp1_atm = kpa_to_atm(P_vp1_kpa)
                 P_vp2_atm = kpa_to_atm(P_vp2_kpa)

                 bubble_P = x1 * gamma1 * P_vp1_atm + (1 - x1) * gamma2 * P_vp2_atm
                 return bubble_P - P_atm
             except:
                 return np.inf # Return a large value if calculation fails

         # Estimate initial temperature range (between boiling points)
         T_guess_low = min(kpa_to_atm(10**antoine_coeffs1[0] - antoine_coeffs1[1]/(P_atm + antoine_coeffs1[2])),
                           kpa_to_atm(10**antoine_coeffs2[0] - antoine_coeffs2[1]/(P_atm + antoine_coeffs2[2]))) # Simplified inverse Antoine
         T_guess_high = max(kpa_to_atm(10**antoine_coeffs1[0] - antoine_coeffs1[1]/(P_atm + antoine_coeffs1[2])),
                            kpa_to_atm(10**antoine_coeffs2[0] - antoine_coeffs2[1]/(P_atm + antoine_coeffs2[2]))) # Simplified inverse Antoine

         # Using root_scalar to find the temperature where pressure_difference is zero
         try:
             # Adjust bracket based on boiling points at the given pressure if possible
             # For now, use a wide bracket or initial guess
             sol = root_scalar(pressure_difference, bracket=[min(bp1, bp2)-20, max(bp1, bp2)+20]) # Using component boiling points as rough guide
             return sol.root
         except Exception as e:
             print(f"Could not find bubble point temperature for x1={x1}, P={P_atm}: {e}")
             return np.nan


     # Calculate the bubble point temperature for the given liquid composition x
     eq_temp_celsius = bubble_point_temperature(x, P_atm, antoine_coeffs1, antoine_coeffs2,
                                               gamma1_func=lambda x_comp, T: gamma1, # Pass the specific activity coef value if constant
                                               gamma2_func=lambda x_comp, T: gamma2) # Pass the specific activity coef value if constant


     if np.isnan(eq_temp_celsius):
         return np.nan # Return NaN if temperature calculation failed

     # Calculate vapor pressure at the equilibrium temperature
     P_vp1_kpa = antoine_vapor_pressure(eq_temp_celsius, *antoine_coeffs1)
     P_vp1_atm = kpa_to_atm(P_vp1_kpa)

     # Calculate equilibrium vapor composition using modified Raoult's Law
     y = (x * gamma1 * P_vp1_atm) / P_atm

     return y


# Modify the plot_mccabe_thiele function to use the new equilibrium function
def plot_mccabe_thiele(antoine_coeffs1, antoine_coeffs2, R, xd, xb, zf, q, F, D, B, P_atm, gamma1=1.0, gamma2=1.0):
    """
    Draw a fully labelled McCabe-Thiele diagram using rigorous VLE and return the figure.
    Uses Antoine equation for vapor pressure and optional activity coefficients.
    """

    def _equilibrium_vle(x, T_celsius, P_atm, antoine_coeffs1, antoine_coeffs2, gamma1, gamma2):
         return calculate_vle(x, T_celsius, P_atm, antoine_coeffs1, antoine_coeffs2, gamma1, gamma2)

    def bubble_point_temperature(x1, P_atm, antoine_coeffs1, antoine_coeffs2, gamma1_func=lambda x, T: 1.0, gamma2_func=lambda x, T: 1.0):
         """
         Calculates the bubble point temperature (°C) for a given liquid composition and pressure.
         x1: Liquid mole fraction of component 1
         P_atm: Total pressure in atm
         antoine_coeffs1: (A, B, C) for component 1
         antoine_coeffs2: (A, B, C) for component 2
         gamma1_func: Function to calculate activity coefficient for component 1 (takes x1, T_celsius)
         gamma2_func: Function to calculate activity coefficient for component 2 (takes x1, T_celsius)
         Returns: Bubble point temperature in °C.
         """
         def pressure_difference(T):
             try:
                 gamma1 = gamma1_func(x1, T)
                 gamma2 = gamma2_func(1 - x1, T) # Assuming activity coef depends on liquid comp of that component

                 P_vp1_kpa = antoine_vapor_pressure(T, *antoine_coeffs1)
                 P_vp2_kpa = antoine_vapor_pressure(T, *antoine_coeffs2)
                 P_vp1_atm = kpa_to_atm(P_vp1_kpa)
                 P_vp2_atm = kpa_to_atm(P_vp2_kpa)

                 bubble_P = x1 * gamma1 * P_vp1_atm + (1 - x1) * gamma2 * P_vp2_atm
                 return bubble_P - P_atm
             except:
                 return np.inf # Return a large value if calculation fails

         # Estimate initial temperature range (between boiling points)
         # More robust initial guess might involve solving for T at x=0 and x=1
         # For now, a simple range or fixed guess might work for many systems
         # Need to get bp1 and bp2 into this scope or pass them
         # Assuming bp1 and bp2 are available or can be calculated from Antoine at 1 atm
         # A more robust approach would be to find the roots of pressure_difference at x1=0 and x1=1
         try:
             # Find boiling point of component 1 at P_atm
             def bp1_eq(T):
                 P_vp1_kpa = antoine_vapor_pressure(T, *antoine_coeffs1)
                 return kpa_to_atm(P_vp1_kpa) - P_atm
             bp1_at_P = root_scalar(bp1_eq, bracket=[0, 300]).root # Assuming reasonable temperature range

             # Find boiling point of component 2 at P_atm
             def bp2_eq(T):
                 P_vp2_kpa = antoine_vapor_pressure(T, *antoine_coeffs2)
                 return kpa_to_atm(P_vp2_kpa) - P_atm
             bp2_at_P = root_scalar(bp2_eq, bracket=[0, 300]).root # Assuming reasonable temperature range

             T_guess_low = min(bp1_at_P, bp2_at_P) - 10 # Add some buffer
             T_guess_high = max(bp1_at_P, bp2_at_P) + 10 # Add some buffer

             sol = root_scalar(pressure_difference, bracket=[T_guess_low, T_guess_high])
             return sol.root
         except Exception as e:
             print(f"Could not find bubble point temperature for x1={x1}, P={P_atm} in plot_mccabe_thiele: {e}")
             return np.nan


    # ------- 1) curves -------------------------------------------------
    x = np.linspace(0, 1, 100) # Reduced points for potentially slower VLE calcs
    y_eq = []
    T_eq = []
    for xi in x:
        if np.isclose(xi, 0):
            # Handle x=0 case: equilibrium vapor is pure heavy key
            y_eq.append(0.0)
            # Temperature is boiling point of heavy key at P_atm
            def bp2_eq(T):
                 P_vp2_kpa = antoine_vapor_pressure(T, *antoine_coeffs2)
                 return kpa_to_atm(P_vp2_kpa) - P_atm
            try:
                T_eq.append(root_scalar(bp2_eq, bracket=[0, 300]).root)
            except:
                T_eq.append(np.nan)

        elif np.isclose(xi, 1):
             # Handle x=1 case: equilibrium vapor is pure light key
            y_eq.append(1.0)
            # Temperature is boiling point of light key at P_atm
            def bp1_eq(T):
                 P_vp1_kpa = antoine_vapor_pressure(T, *antoine_coeffs1)
                 return kpa_to_atm(P_vp1_kpa) - P_atm
            try:
                T_eq.append(root_scalar(bp1_eq, bracket=[0, 300]).root)
            except:
                T_eq.append(np.nan)

        else:
            # Calculate bubble point temperature for each liquid composition
            eq_temp_celsius = bubble_point_temperature(xi, P_atm, antoine_coeffs1, antoine_coeffs2,
                                                       gamma1_func=lambda x_comp, T: gamma1,
                                                       gamma2_func=lambda x_comp, T: gamma2)
            T_eq.append(eq_temp_celsius)

            if not np.isnan(eq_temp_celsius):
                # Calculate equilibrium vapor composition at this temperature
                y_eq.append(_equilibrium_vle(xi, eq_temp_celsius, P_atm, antoine_coeffs1, antoine_coeffs2, gamma1, gamma2))
            else:
                y_eq.append(np.nan) # Append NaN if temperature calculation failed


    y_eq = np.array(y_eq)
    valid_indices = ~np.isnan(y_eq)
    x = x[valid_indices]
    y_eq = y_eq[valid_indices]


    y_rect = _rectifying(x, R, xd)
    y_strip = _stripping(x, R, q, F, D, B, xb)

    # q-line intersection with equilibrium
    qx = np.linspace(0, 1, 2)
    qy = _q_line(qx, q, zf)

    # ------- 2) staircase construction ---------------------------------
    stairs_x, stairs_y = [xd], [xd]
    feed_found = False
    feed_stage = None
    n_stages = 0

    x_op, y_op = xd, xd  # start on operating line (xD, xD)

    # Need a way to get equilibrium y for a given x using the rigorous model
    # This requires finding the bubble point temperature for x and P, then calculating y.
    # Let's create an interpolation function for the equilibrium curve generated above.
    from scipy.interpolate import interp1d
    # Ensure x and y_eq are monotonically increasing for interpolation
    sort_indices = np.argsort(x)
    x_sorted = x[sort_indices]
    y_eq_sorted = y_eq[sort_indices]

    # Handle potential duplicate x values or issues from sorting NaNs if any slipped through
    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    y_eq_unique = y_eq_sorted[unique_indices]

    # If there are still issues with unique points, fall back to a simple connection or error
    if len(unique_x) < 2:
         print("Warning: Could not create a valid equilibrium curve for stepping stages.")
         # Fallback or error handling needed
         return plt.figure(), plt.gca(), 0, None # Return empty plot or handle appropriately


    # Need to handle cases where interpolation might go out of bounds or fail
    # Use fill_value="extrapolate" with caution or clip x_op
    # Let's try clipping x_op to the range of unique_x
    interp_kind = 'linear' # Use linear interpolation for simplicity

    # Need to handle the case where y_eq_unique is not monotonically increasing
    # if not np.all(np.diff(y_eq_unique) >= 0):
    #     print("Warning: Equilibrium curve is not monotonically increasing. Interpolation may be inaccurate.")
        # Could attempt smoothing or other methods, but for now proceed with caution

    try:
        equilibrium_interp = interp1d(unique_x, y_eq_unique, kind=interp_kind, fill_value="extrapolate")
    except Exception as e:
        print(f"Error creating interpolation function: {e}")
        # Fallback or error handling needed
        return plt.figure(), plt.gca(), 0, None # Return empty plot or handle appropriately


    while x_op > xb and n_stages < 100:
        # Ensure x_op is within the range of the interpolation data
        x_op_clipped = np.clip(x_op, unique_x.min(), unique_x.max())

        try:
            y_eq_pt = equilibrium_interp(x_op_clipped).item() # Use .item() to get scalar from numpy array
        except Exception as e:
             print(f"Error during equilibrium interpolation at x_op={x_op}: {e}")
             break # Break if interpolation fails

        stairs_x.extend([x_op, x_op_clipped]) # Use original x_op for horizontal step start
        stairs_y.extend([y_op, y_eq_pt])

        # vertical to operating line - need to determine which operating line based on feed stage
        # This still relies on the q-line intersection to determine the feed stage transition
        # Need the intersection point of the q-line and rectifying line to switch operating lines.

        # Intersection of q-line (y = q/(q-1) * x - zf/(q-1)) and rectifying line (y = R/(R+1) * x + xd/(R+1))
        # q/(q-1) * x - zf/(q-1) = R/(R+1) * x + xd/(R+1)
        # (q/(q-1) - R/(R+1)) * x = xd/(R+1) + zf/(q-1)
        # x_intersect = (xd/(R+1) + zf/(q-1)) / (q/(q-1) - R/(R+1))

        # Handle q=1 case (vertical q-line at x=zf)
        if abs(q - 1.0) < 1e-6:
            x_intersect = zf
        elif abs((q/(q-1) - R/(R+1))) < 1e-9: # Avoid division by zero if slopes are equal
             x_intersect = np.nan # Parallel lines, no single intersection point relevant for stage stepping
        else:
            x_intersect = (xd/(R+1) + zf/(q-1)) / (q/(q-1) - R/(R+1))

        # Determine which operating line to use
        if not feed_found and x_op_clipped > x_intersect:
            # In rectifying section
            # Find x_next on the rectifying line
            x_next = (y_eq_pt - xd/(R+1)) * (R+1)/R
            y_next = _rectifying(x_next, R, xd)
        else:
            # In stripping section (or at/below intersection)
            if not feed_found:
                feed_found = True
                feed_stage = n_stages + 1 # Feed is on the first stage below the intersection
            # Find x_next on the stripping line
            Lbar = R * D + q * F
            Vbar = Lbar - B
            # Avoid division by zero for Vbar
            if abs(Vbar) < 1e-9:
                 print("Warning: Vbar is close to zero. Cannot calculate stripping line.")
                 break # Cannot continue stage stepping
            x_next = (y_eq_pt + B * xb / Vbar) * Vbar / Lbar
            y_next = _stripping(x_next, R, q, F, D, B, xb) # Recalculate y_next on the stripping line


        stairs_x.extend([x_op_clipped, x_next])
        stairs_y.extend([y_eq_pt, y_next])

        x_op, y_op = x_next, y_next
        n_stages += 1
        if x_op < xb:
            break
        if n_stages > 200: # Safety break for complex VLE
            print("Warning: Reached maximum stages (200). Stopping stage calculation.")
            break


    # ------- 3) plotting ------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x, y_eq, 'b', lw=2, label='Equilibrium (Rigorous VLE)')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.7)

    # Operating lines
    ax.plot(np.linspace(x_intersect if not np.isnan(x_intersect) else 0, xd, 100),
            _rectifying(np.linspace(x_intersect if not np.isnan(x_intersect) else 0, xd, 100), R, xd),
            'g', lw=2, label='Rectifying')

    # Need to determine the x range for the stripping line plot more accurately
    # It goes from xb to the intersection point
    ax.plot(np.linspace(xb, x_intersect if not np.isnan(x_intersect) else 1, 100),
            _stripping(np.linspace(xb, x_intersect if not np.isnan(x_intersect) else 1, 100), R, q, F, D, B, xb),
            'r', lw=2, label='Stripping')


    # q-line
    if abs(q - 1.0) < 1e-6: # Handle q = 1 (saturated liquid)
        ax.axvline(x=zf, color='m', linestyle='--', lw=2, label='q-line')
    elif abs(q - 1.0) > 1e-3:
        ax.plot(qx, qy, 'm--', lw=2, label='q-line')


    # staircase
    ax.plot(stairs_x, stairs_y, 'o-', color='orange', markersize=4,
            label='Stages')

    # highlight feed stage
    if feed_stage is not None and feed_stage * 4 < len(stairs_x): # Ensure index is valid
        idx = feed_stage * 4 # Indexing into the flattened stairs_x/y list (x0, y0, x1, y1, x2, y2...)
        # Need to check if this index is correct for the stage transition point
        # The staircase is xd, y_eq_xd, x_op1, y_op1, x_op1, y_eq_op1, x_op2, y_op2...
        # Stage 1 is (xd, y_eq_xd) to (x_op1, y_op1)
        # Stage i is (x_opi-1, y_eq_opi-1) to (x_opi, y_opi)
        # The points on the equilibrium curve are at indices 1, 5, 9, ... (4i - 3)
        # The points on the operating line are at indices 3, 7, 11, ... (4i - 1)
        # The feed stage transition happens *after* a point on the equilibrium curve (y_eq_pt)
        # and *before* the next point on the operating line (y_next).
        # The x value where the transition happens is x_op_clipped (or x_op if not clipped)
        # The y value where the transition happens is y_eq_pt
        # The feed stage is the stage *below* the feed point.
        # If feed_stage is n, the feed point is between stage n and n-1.
        # The staircase point *before* the feed stage transition is the point on the equilibrium curve.
        # The index of this point in stairs_x/y is 4 * (feed_stage - 1) + 1
        # Let's try plotting the point on the equilibrium curve just before the feed stage transition.
        # This assumes feed_stage is 1-indexed number of stages above the feed.
        # If feed_stage is the stage *where* the feed enters, then it's the point *on* that stage.
        # Based on the calculation `feed_stage = n_stages + 1`, it seems `n_stages` is the number of stages *above* the feed.
        # So feed stage is the (n_stages + 1)-th stage from the top.
        # The point on the equilibrium curve for stage n is at index 4*(n-1)+1 in stairs_x/y
        # So for feed stage (n_stages+1), the point is at index 4*(n_stages)+1
        # Let's use the calculated intersection point (x_intersect, y_intersect_rect) to highlight the feed location on the plot instead.
        if not np.isnan(x_intersect):
            y_intersect_rect = _rectifying(x_intersect, R, xd)
            ax.scatter(x_intersect, y_intersect_rect, color='purple',
                       s=80, zorder=5, label=f'Feed Intersection (x={x_intersect:.2f})')
        else:
             print("Warning: Could not plot feed intersection point due to invalid calculation.")


    # stage numbers (every 2nd point on equilibrium curve)
    # Stage numbers are 1-indexed from the top.
    # The equilibrium points are at indices 1, 5, 9, ... (4*i - 3) for stage i
    for i in range(1, n_stages + 1):
        idx = 4 * (i-1) + 1 # Index of the point on the equilibrium curve for stage i
        if idx < len(stairs_x):
            # Position the text near the step
             text_x = stairs_x[idx]
             text_y = stairs_y[idx]
             # Adjust text position slightly for better readability
             ax.text(text_x + 0.01, text_y + 0.01, str(i),
                    fontsize=9, color='navy')


    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Liquid mole fraction (light key)')
    ax.set_ylabel('Vapor mole fraction (light key)')
    ax.set_title('McCabe-Thiele Diagram (Rigorous VLE)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, n_stages, feed_stage # feed_stage from staircase construction is approximate


def calculate_feed_stage(antoine_coeffs1, antoine_coeffs2, R, xd, xb, zf, q, F_mol, D, B, P_atm, gamma1=1.0, gamma2=1.0):
    """
    Calculates the estimated feed stage based on the intersection of the
    q-line and the rigorous operating lines.
    """
    # Calculate intersection of q-line and rectifying line
    # q-line: y = q/(q-1) * x - zf/(q-1)
    # Rectifying line: y = R/(R+1) * x + xd/(R+1)

    # Handle q=1 case (vertical q-line at x=zf)
    if abs(q - 1.0) < 1e-6:
        x_feed_intersect = zf
    elif abs((q/(q-1) - R/(R+1))) < 1e-9: # Avoid division by zero if slopes are equal
         x_feed_intersect = np.nan # Parallel lines
    else:
        try:
            # Solving q/(q-1) * x - zf/(q-1) = R/(R+1) * x + xd/(R+1) for x
            # x * (q/(q-1) - R/(R+1)) = xd/(R+1) + zf/(q-1)
            x_feed_intersect = (xd/(R+1) + zf/(q-1)) / (q/(q-1) - R/(R+1))
        except Exception as e:
            print(f"Error calculating q-line/rectifying line intersection: {e}")
            x_feed_intersect = np.nan


    if np.isnan(x_feed_intersect):
        return None # Cannot determine feed stage if intersection is undefined

    # Now, find the stage on the McCabe-Thiele diagram where the liquid composition
    # crosses this intersection x-coordinate.
    # This requires performing the stage-by-stage calculation and checking where the liquid x
    # goes from > x_feed_intersect to <= x_feed_intersect.

    # Recalculate stages with the rigorous VLE
    # Need the stage calculation function to be compatible with rigorous VLE

    def calculate_stages_rigorous(xd, xb, zf, q, R, F_mol, D, B, P_atm, antoine_coeffs1, antoine_coeffs2, gamma1=1.0, gamma2=1.0):
        """
        Calculates theoretical stages using rigorous VLE for McCabe-Thiele.
        Returns: x_stages, y_stages, number of stages, and estimated feed stage.
        """
        def _equilibrium_rigorous(x):
             # Find bubble point temperature for liquid composition x
             eq_temp_celsius = bubble_point_temperature(x, P_atm, antoine_coeffs1, antoine_coeffs2,
                                                       gamma1_func=lambda x_comp, T: gamma1,
                                                       gamma2_func=lambda x_comp, T: gamma2)
             if np.isnan(eq_temp_celsius):
                 return np.nan # Return NaN if temperature calculation failed

             # Calculate equilibrium vapor composition at this temperature
             return calculate_vle(x, eq_temp_celsius, P_atm, antoine_coeffs1, antoine_coeffs2, gamma1, gamma2)

        def _rectifying_line_func(x):
            return R/(R+1) * x + xd/(R+1)

        def _stripping_line_func(x):
            L_bar = R * D + q * F_mol
            V_bar = L_bar - B
            # Avoid division by zero for V_bar
            if abs(V_bar) < 1e-9:
                 print("Warning: Vbar is close to zero in stage calculation. Cannot calculate stripping line.")
                 return np.full_like(x, np.nan)
            return (L_bar / V_bar) * x - (B * xb) / V_bar

        x_stages = [xd]
        y_stages = [xd]
        x_current = xd
        stage_count = 0
        feed_stage_est = None # Estimated feed stage based on intersection

        # Calculate intersection point for switching operating lines
        # This is the same x_feed_intersect calculated earlier
        # Need the corresponding y value on the rectifying line at this x
        y_feed_intersect_rect = _rectifying_line_func(x_feed_intersect) if not np.isnan(x_feed_intersect) else np.nan


        while x_current > xb and stage_count < 200: # Safety break
            y_eq = _equilibrium_rigorous(x_current)

            if np.isnan(y_eq):
                 print(f"Warning: Equilibrium calculation failed at x={x_current:.2f}. Stopping stage calculation.")
                 break

            x_stages.append(x_current)
            y_stages.append(y_eq)

            # Determine which operating line to step down to
            # Switch from rectifying to stripping when y_eq <= y_feed_intersect_rect AND liquid x_current <= x_feed_intersect
            # This condition for switching needs careful consideration based on the q-line
            # A simpler approach for stepping is to switch when the current liquid composition x_current
            # is less than or equal to the feed composition zf, and the stage number is >= the estimated feed stage number from intersection.
            # However, the McCabe-Thiele stepping logic is based on which side of the q-line the equilibrium point lies.
            # Let's stick to the logic based on the q-line intersection with the operating lines.
            # The feed stage is where the liquid on the stage has a composition x such that
            # the equilibrium vapor y is on the stripping line, and the stage above had an equilibrium vapor y
            # that was on the rectifying line.

            # Let's use the feed composition zf and q-line to determine the section switch for stepping.
            # The switch happens when the equilibrium point (x_current, y_eq) crosses the q-line.
            # The q-line equation: y = q/(q-1) * x - zf/(q-1)

            # Handle q=1 case separately for the q-line check
            if abs(q - 1.0) < 1e-6: # Saturated liquid feed, vertical q-line at x = zf
                 on_stripping_side = (x_current <= zf)
            elif abs(q - 0.0) < 1e-6: # Saturated vapor feed, horizontal q-line at y = zf
                 on_stripping_side = (y_eq <= zf)
            elif abs(q - 1.0) > 1e-3: # Other feed conditions
                try:
                    y_q_line = _q_line(x_current, q, zf)
                    on_stripping_side = (y_eq <= y_q_line) # Check if equilibrium point is below or on the q-line
                except: # Handle cases where q-line calculation might fail (e.g., q close to 1)
                    on_stripping_side = False # Default to rectifying if q-line is problematic
            else: # q = 0 (Saturated Vapor)
                 on_stripping_side = (y_eq <= zf) # Horizontal q-line at y = zf


            # Determine which operating line to use for the next step (finding x_next from y_eq)
            # If the current stage is in the rectifying section (above the feed), use the rectifying line equation to find the liquid composition of the stage below.
            # If the current stage is in the stripping section (at or below the feed), use the stripping line equation.

            # The switch between operating lines for stepping occurs at the feed stage.
            # The feed stage is where the liquid composition x on the tray is such that
            # the equilibrium vapor composition y_eq is on the stripping line when coming from the rectifying line.

            # A simpler way to implement the stepping logic for McCabe-Thiele is to check if the current liquid composition x_current
            # is greater than the feed composition at the intersection point (x_feed_intersect).
            # If x_current > x_feed_intersect, we are in the rectifying section.
            # If x_current <= x_feed_intersect, we are in the stripping section.

            # Need to be careful about the first stage (condenser) which is always on the rectifying line.
            if stage_count == 0 or x_current > x_feed_intersect:
                # Use rectifying line to find x of the stage below (stepping down)
                # y_eq = R/(R+1) * x_next + xd/(R+1)
                # x_next = (y_eq - xd/(R+1)) * (R+1)/R
                 x_next = (y_eq - xd/(R+1)) * (R+1)/R
                 y_next_op = _rectifying_line_func(x_next) # Corresponding y on the rectifying line

            else:
                # Use stripping line to find x of the stage below (stepping down)
                # y_eq = (L_bar / V_bar) * x_next - (B * xb) / V_bar
                # (y_eq + (B * xb) / V_bar) = (L_bar / V_bar) * x_next
                # x_next = (y_eq + (B * xb) / V_bar) * (V_bar / L_bar)
                L_bar = R * D + q * F_mol
                V_bar = L_bar - B
                if abs(L_bar) < 1e-9:
                     print("Warning: Lbar is close to zero in stage calculation. Cannot calculate stripping line step.")
                     break
                x_next = (y_eq + (B * xb) / V_bar) * (V_bar / L_bar)
                y_next_op = _stripping_line_func(x_next) # Corresponding y on the stripping line

            # Check if we just crossed the feed point x-composition to estimate feed stage
            if feed_stage_est is None and x_current > x_feed_intersect and x_next <= x_feed_intersect:
                 feed_stage_est = stage_count + 1 # Feed is on the next stage (stage_count + 1)

            x_stages.append(x_next)
            y_stages.append(y_next_op) # Append the y value on the operating line

            x_current = x_next
            stage_count += 1

            # Add a check to prevent stepping backwards or infinite loops if VLE is complex
            if stage_count > 1 and x_stages[-1] >= x_stages[-3]:
                 print(f"Warning: Stage calculation not progressing towards xb. Stopping at stage {stage_count}.")
                 break


        return x_stages, y_stages, stage_count, feed_stage_est

    # Call the rigorous stage calculation within plot_mccabe_thiele
    # This will generate the actual points for the staircase using rigorous VLE and stepping logic
    # and also provide an estimated feed stage from the stepping process.

    # Need to regenerate the stairs_x, stairs_y, n_stages, and feed_stage based on the rigorous calculation.
    # The initial staircase construction code in plot_mccabe_thiele was based on the old, simplified equilibrium.
    # Let's replace it with the call to the new rigorous stage calculation.

    x_stages, y_stages, n_stages, feed_stage_rigorous = calculate_stages_rigorous(
         xd, xb, zf, q, R, F, D, B, P_atm, antoine_coeffs1, antoine_coeffs2, gamma1, gamma2
    )

    # The stairs_x and stairs_y from calculate_stages_rigorous have a different structure
    # They are [xd, xd, x_stage1_L, y_stage1_V, x_stage1_L, y_stage1_L_op, x_stage2_L, y_stage2_V...]
    # We need to format them to match the stairs_x, stairs_y structure expected by the plotting code:
    # [xd, xd, x_stage1_L, y_eq_stage1_L, x_stage1_L, y_op_stage1_L, x_stage2_L, y_eq_stage2_L, x_stage2_L, y_op_stage2_L ...]
    # The calculate_stages_rigorous function returns [x_liquid_on_tray_i, y_vapor_from_tray_i, x_liquid_on_tray_i+1, y_liquid_on_tray_i_op]
    # Let's adjust calculate_stages_rigorous to return the points in the desired format for plotting.

    # Revised calculate_stages_rigorous to return points in the plotting format:
    def calculate_stages_rigorous_plotting(xd, xb, zf, q, R, F_mol, D, B, P_atm, antoine_coeffs1, antoine_coeffs2, gamma1=1.0, gamma2=1.0):
        """
        Calculates theoretical stages using rigorous VLE for McCabe-Thiele plotting.
        Returns: stairs_x, stairs_y lists suitable for plotting, number of stages, and estimated feed stage.
        """
        def _equilibrium_rigorous(x):
             eq_temp_celsius = bubble_point_temperature(x, P_atm, antoine_coeffs1, antoine_coeffs2,
                                                       gamma1_func=lambda x_comp, T: gamma1,
                                                       gamma2_func=lambda x_comp, T: gamma2)
             if np.isnan(eq_temp_celsius):
                 return np.nan
             return calculate_vle(x, eq_temp_celsius, P_atm, antoine_coeffs1, antoine_coeffs2, gamma1, gamma2)

        def _rectifying_line_func(x):
            return R/(R+1) * x + xd/(R+1)

        def _stripping_line_func(x):
            L_bar = R * D + q * F_mol
            V_bar = L_bar - B
            if abs(V_bar) < 1e-9:
                 return np.nan
            return (L_bar / V_bar) * x - (B * xb) / V_bar

        stairs_x = [xd]
        stairs_y = [xd]
        x_current = xd # Liquid composition leaving the current stage (starts at condenser, x=xd)
        stage_count = 0
        feed_stage_est = None

        # Calculate intersection point for operating line switch
        if abs(q - 1.0) < 1e-6: x_feed_intersect = zf
        elif abs((q/(q-1) - R/(R+1))) < 1e-9: x_feed_intersect = np.nan
        else:
            try:
                x_feed_intersect = (xd/(R+1) + zf/(q-1)) / (q/(q-1) - R/(R+1))
            except:
                x_feed_intersect = np.nan


        while x_current > xb and stage_count < 200:
            # Point on equilibrium curve from x_current (liquid leaving stage n)
            y_eq_from_x_current = _equilibrium_rigorous(x_current)

            if np.isnan(y_eq_from_x_current):
                 print(f"Warning: Equilibrium calculation failed at x={x_current:.2f} during plotting stages. Stopping.")
                 break

            # Add horizontal step to equilibrium curve
            stairs_x.append(x_current)
            stairs_y.append(y_eq_from_x_current)

            # Determine which operating line to use to find the liquid composition of the stage below (x_next)
            # If x_current is > x_feed_intersect, we are in the rectifying section above the feed.
            # If x_current <= x_feed_intersect, we are in the stripping section at or below the feed.
            # Special case for the first step from xd: it's always on the rectifying line.
            # The switch happens when the liquid composition *leaving* the stage crosses the feed intersection point.

            if stage_count == 0 or x_current > x_feed_intersect:
                 # Use rectifying line to find x_next (liquid leaving the stage below)
                 # y_eq_from_x_current = R/(R+1) * x_next + xd/(R+1)
                 # x_next = (y_eq_from_x_current - xd/(R+1)) * (R+1)/R
                 x_next = (y_eq_from_x_current - xd/(R+1)) * (R+1)/R
                 # Add vertical step down to the operating line
                 stairs_x.append(x_next)
                 stairs_y.append(y_eq_from_x_current)

                 # Add point on the operating line at x_next
                 stairs_x.append(x_next)
                 stairs_y.append(_rectifying_line_func(x_next)) # This is the y value on the rectifying line at x_next


            else:
                 # Use stripping line to find x_next (liquid leaving the stage below)
                 # y_eq_from_x_current = (L_bar / V_bar) * x_next - (B * xb) / V_bar
                 L_bar = R * D + q * F_mol
                 V_bar = L_bar - B
                 if abs(L_bar) < 1e-9:
                      print("Warning: Lbar is close to zero in plotting stage calculation. Cannot calculate stripping line step.")
                      break
                 x_next = (y_eq_from_x_current + (B * xb) / V_bar) * (V_bar / L_bar)
                 # Add vertical step down to the operating line
                 stairs_x.append(x_next)
                 stairs_y.append(y_eq_from_x_current)

                 # Add point on the operating line at x_next
                 stairs_x.append(x_next)
                 stairs_y.append(_stripping_line_func(x_next)) # This is the y value on the stripping line at x_next


            # Check if this stage is the feed stage
            # The feed stage is where the liquid composition leaving the stage (x_current)
            # is just above the feed intersection x_feed_intersect, and the liquid composition
            # on the stage below (x_next) is below or at x_feed_intersect.
            # This means the feed is introduced to the stage where the liquid composition is x_next.
            # The feed stage is the stage *below* the one where x_current > x_feed_intersect and x_next <= x_feed_intersect.
            # If stage_count is the current stage number (starting from 0 for condenser), then the liquid leaving this stage is x_current.
            # The liquid entering this stage from below is the vapor from stage stage_count+1.
            # The feed enters stage_count + 1 if x_current > x_feed_intersect and x_next <= x_feed_intersect.
            # The feed stage number is stage_count + 1 (1-indexed from top).

            if feed_stage_est is None and x_current > x_feed_intersect and x_next <= x_feed_intersect:
                 feed_stage_est = stage_count + 1 # Feed is on the next stage (stage_count + 1)

            x_current = x_next
            stage_count += 1

            # Add a check to prevent stepping backwards or infinite loops
            if stage_count > 1 and stairs_x[-1] >= stairs_x[-5]: # Compare current x_next with x_current from two steps ago
                 print(f"Warning: Stage calculation not progressing towards xb. Stopping at stage {stage_count}.")
                 break

        return stairs_x, stairs_y, stage_count, feed_stage_est


    # Call the revised rigorous stage calculation for plotting
    stairs_x, stairs_y, n_stages, feed_stage_rigorous = calculate_stages_rigorous_plotting(
         xd, xb, zf, q, R, F, D, B, P_atm, antoine_coeffs1, antoine_coeffs2, gamma1, gamma2
    )

    # ------- 3) plotting (remains mostly the same, uses the new stairs_x/y) ------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot the rigorous equilibrium curve (recalculate with more points for smooth curve)
    x_eq_plot = np.linspace(0, 1, 400)
    y_eq_plot = []
    for xi_plot in x_eq_plot:
         if np.isclose(xi_plot, 0):
              y_eq_plot.append(0.0)
         elif np.isclose(xi_plot, 1):
              y_eq_plot.append(1.0)
         else:
              eq_temp_celsius = bubble_point_temperature(xi_plot, P_atm, antoine_coeffs1, antoine_coeffs2,
                                                       gamma1_func=lambda x_comp, T: gamma1,
                                                       gamma2_func=lambda x_comp, T: gamma2)
              if not np.isnan(eq_temp_celsius):
                   y_eq_plot.append(_equilibrium_vle(xi_plot, eq_temp_celsius, P_atm, antoine_coeffs1, antoine_coeffs2, gamma1, gamma2))
              else:
                   y_eq_plot.append(np.nan)

    y_eq_plot = np.array(y_eq_plot)
    valid_indices_plot = ~np.isnan(y_eq_plot)
    x_eq_plot = x_eq_plot[valid_indices_plot]
    y_eq_plot = y_eq_plot[valid_indices_plot]


    ax.plot(x_eq_plot, y_eq_plot, 'b', lw=2, label='Equilibrium (Rigorous VLE)')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.7)

    # Operating lines - plot from the intersection point
    if not np.isnan(x_feed_intersect):
         ax.plot(np.linspace(x_feed_intersect, xd, 100),
                 _rectifying(np.linspace(x_feed_intersect, xd, 100), R, xd),
                 'g', lw=2, label='Rectifying')

         ax.plot(np.linspace(xb, x_feed_intersect, 100),
                 _stripping(np.linspace(xb, x_feed_intersect, 100), R, q, F, D, B, xb),
                 'r', lw=2, label='Stripping')
    else: # Fallback plotting if intersection is NaN
         ax.plot(np.linspace(0, xd, 100), _rectifying(np.linspace(0, xd, 100), R, xd), 'g', lw=2, label='Rectifying')
         ax.plot(np.linspace(xb, 1, 100), _stripping(np.linspace(xb, 1, 100), R, q, F, D, B, xb), 'r', lw=2, label='Stripping')


    # q-line
    if abs(q - 1.0) < 1e-6: # Handle q = 1 (saturated liquid)
        ax.axvline(x=zf, color='m', linestyle='--', lw=2, label='q-line')
    elif abs(q - 1.0) > 1e-3:
        qx_plot = np.linspace(0, 1, 2)
        qy_plot = _q_line(qx_plot, q, zf)
        ax.plot(qx_plot, qy_plot, 'm--', lw=2, label='q-line')


    # staircase
    ax.plot(stairs_x, stairs_y, 'o-', color='orange', markersize=4,
            label='Stages')

    # highlight feed stage - use the calculated feed stage from the rigorous stepping
    if feed_stage_rigorous is not None and feed_stage_rigorous > 0: # Ensure feed_stage_rigorous is valid and > 0
        # Find the liquid composition on the feed stage
        # The liquid composition leaving stage N is at index 4*(N-1) in stairs_x/y
        feed_stage_liquid_x_index = 4 * (feed_stage_rigorous - 1)
        if feed_stage_liquid_x_index < len(stairs_x):
             feed_stage_liquid_x = stairs_x[feed_stage_liquid_x_index]
             # Find the corresponding vapor composition from that stage (on the equilibrium curve)
             feed_stage_vapor_y_index = feed_stage_liquid_x_index + 1
             if feed_stage_vapor_y_index < len(stairs_y):
                 feed_stage_vapor_y = stairs_y[feed_stage_vapor_y_index]
                 ax.scatter(feed_stage_liquid_x, feed_stage_vapor_y, color='purple',
                            s=80, zorder=5, label=f'Feed stage ({feed_stage_rigorous})')
             else:
                  print(f"Warning: Could not mark feed stage {feed_stage_rigorous} on plot due to invalid index.")
        else:
             print(f"Warning: Could not find liquid composition for feed stage {feed_stage_rigorous} on plot.")


    # stage numbers (every 2nd point on equilibrium curve)
    for i in range(1, n_stages + 1):
        idx = 4 * (i-1) + 1 # Index of the point on the equilibrium curve for stage i
        if idx < len(stairs_x):
             # Position the text near the step
             text_x = stairs_x[idx]
             text_y = stairs_y[idx]
             # Adjust text position slightly for better readability
             ax.text(text_x + 0.01, text_y + 0.01, str(i),
                    fontsize=9, color='navy')


    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Liquid mole fraction (light key)')
    ax.set_ylabel('Vapor mole fraction (light key)')
    ax.set_title('McCabe-Thiele Diagram (Rigorous VLE)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, n_stages, feed_stage_rigorous # Return rigorous feed stage estimate


# The calculate_stages function is no longer used for plotting, but might be used elsewhere?
# Let's keep it for now but rename it or clarify its purpose if needed.
# The original calculate_stages seems to be for the staircase construction logic itself,
# which has now been integrated into plot_mccabe_thiele using the rigorous VLE.
# Let's remove the old calculate_stages function to avoid confusion.

# Remove the old calculate_stages function definition from here...
# def calculate_stages(alpha, R, xd, xb, feed_comp, q_value, F_mol, D, B):
# ... (rest of the old function)


# Update calculate_feed_stage to use the rigorous stage calculation result
def calculate_feed_stage(antoine_coeffs1, antoine_coeffs2, R, xd, xb, zf, q, F_mol, D, B, P_atm, gamma1=1.0, gamma2=1.0):
    """
    Calculates the estimated feed stage based on the rigorous stage calculation.
    This function now calls the rigorous stage calculation and returns its feed stage estimate.
    """
    # The logic for estimating the feed stage based on the intersection of operating lines
    # is still valid, but the rigorous stage calculation also provides an estimate.
    # Let's return the estimate from the rigorous stage calculation for consistency with the plot.

    stairs_x, stairs_y, n_stages, feed_stage_rigorous = calculate_stages_rigorous_plotting(
         xd, xb, zf, q, R, F_mol, D, B, P_atm, antoine_coeffs1, antoine_coeffs2, gamma1, gamma2
    )

    return feed_stage_rigorous # Return the feed stage estimated during rigorous stepping


# The main function from the original script is not used in the Streamlit app context.
# Remove the main function definition.
# def main():
# ... (rest of the main function)

# Remove the if __name__ == "__main__": block.
# if __name__ == "__main__":
#     main()
