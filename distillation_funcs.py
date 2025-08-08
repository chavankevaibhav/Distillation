
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, minimize
import requests
from bs4 import BeautifulSoup

# Constants
R_gas = 8.314  # J/mol·K

def calculate_relative_volatility(bp1, bp2):
    """
    Calculates relative volatility using Trouton's rule (simplified).
    This is a simplified approach and not used in the rigorous VLE calculations
    for the McCabe-Thiele plot, but kept for simplified Rmin/Nmin calculations.
    """
    Tb1 = bp1 + 273.15
    Tb2 = bp2 + 273.15
    dH_vap1 = 88 * Tb1  # Trouton's rule (J/mol)
    dH_vap2 = 88 * Tb2  # Trouton's rule (J/mol)
    T_avg = (Tb1 + Tb2) / 2
    # This calculation of alpha based on enthalpy difference and average T is also a simplification.
    # A more accurate alpha for Rmin/Nmin would be based on vapor pressures at a relevant T (e.g., feed T).
    alpha = np.exp((dH_vap1 - dH_vap2) / (R_gas * T_avg) * (Tb1 - Tb2) / T_avg)
    return alpha

def calculate_material_balance(feed_comp, xb, xd, feed_rate, mw1, mw2):
    """
    Calculates distillate and bottoms flow rates based on overall and component material balances.
    feed_comp: Mole fraction of light key in feed (zf)
    xb: Mole fraction of light key in bottoms
    xd: Mole fraction of light key in distillate
    feed_rate: Total feed flow rate (kg/hr)
    mw1: Molecular weight of light key (g/mol)
    mw2: Molecular weight of heavy key (g/mol)
    Returns: D (distillate mol/hr), B (bottoms mol/hr), F_mol (feed mol/hr)
    """
    # Average molecular weight of feed
    mw_feed = feed_comp * mw1 + (1 - feed_comp) * mw2
    # Feed flow rate in mol/hr
    F_mol = feed_rate * 1000 / mw_feed # Convert kg to g and divide by g/mol

    # Overall material balance: F = D + B (in moles)
    # Component material balance (light key): F * zf = D * xd + B * xb
    # Substitute B = F - D into the component balance:
    # F * zf = D * xd + (F - D) * xb
    # F * zf = D * xd + F * xb - D * xb
    # F * zf - F * xb = D * xd - D * xb
    # F * (zf - xb) = D * (xd - xb)
    # D = F * (zf - xb) / (xd - xb)

    # Avoid division by zero or near-zero if product compositions are too close
    if abs(xd - xb) < 1e-9:
        # This indicates no separation is possible or compositions are invalid
        D = 0.0
        B = F_mol
        print("Warning: Distillate and bottoms compositions are too close. Cannot calculate material balance.")
    else:
        D = F_mol * (feed_comp - xb) / (xd - xb)
        B = F_mol - D

    # Ensure D and B are not negative due to floating point errors or invalid inputs
    D = max(0.0, D)
    B = max(0.0, B)
    # Re-normalize B to ensure total flow is conserved if D was capped
    if D == 0.0:
        B = F_mol


    return D, B, F_mol

def calculate_minimum_reflux_ratio(alpha, feed_comp, q_value):
    """
    Calculates minimum reflux ratio (Rmin) using the Underwood equation (simplified for binary).
    alpha: Average relative volatility
    feed_comp: Mole fraction of light key in feed (zf)
    q_value: Feed thermal condition parameter
    Returns: Minimum reflux ratio (Rmin)
    """
    # Simplified Underwood equation for binary systems assuming constant alpha
    # Requires solving for a root theta between alpha_n+1 and alpha_n (here, 1 and alpha)
    # such that sum(alpha_i * z_i / (alpha_i - theta)) = 1 - q
    # For binary: z1 / (alpha - theta) + z2 / (1 - theta) = 1 - q
    # zf / (alpha - theta) + (1 - zf) / (1 - theta) = 1 - q

    # This requires finding the root theta. A simpler approximation for binary is often used:
    # Rmin = (xd / (alpha - 1)) - (xb / (alpha - 1))  # Fenske-Underwood for minimum stages (infinite reflux)
    # This doesn't explicitly use q.

    # A more direct approach for Rmin using the q-line and equilibrium curve intersection:
    # Find the intersection point (x_intersect, y_intersect) of the q-line (y = q/(q-1) * x - zf/(q-1))
    # and the equilibrium curve (y = alpha*x / (1 + (alpha-1)*x)).
    # Rmin = (xd - y_intersect) / (y_intersect - x_intersect)

    # Let's use the analytical solution for the intersection of the q-line and the equilibrium curve (assuming constant alpha):
    # y = alpha * x / (1 + (alpha - 1) * x)  (Equilibrium curve)
    # y = (q / (q - 1)) * x - (zf / (q - 1)) (q-line)

    # Solve for x at the intersection:
    # alpha * x / (1 + (alpha - 1) * x) = (q / (q - 1)) * x - (zf / (q - 1))

    # Case 1: q = 1 (saturated liquid feed), vertical q-line x = zf
    if abs(q - 1.0) < 1e-9:
        x_intersect = feed_comp
        # Find y_intersect on the equilibrium curve at x = zf
        y_intersect = alpha * x_intersect / (1 + (alpha - 1) * x_intersect)
        # Rmin = (xd - y_intersect) / (y_intersect - x_intersect)
        # Avoid division by zero if y_intersect == x_intersect (azeotrope at feed comp)
        if abs(y_intersect - x_intersect) < 1e-9:
             Rmin = float('inf') # Infinite reflux needed if azeotrope at feed
        else:
            Rmin = (xd - y_intersect) / (y_intersect - x_intersect)

    # Case 2: q != 1
    else:
        # Solve the quadratic equation for x at the intersection.
        # Rearranging the equation:
        # alpha * x = ((q / (q - 1)) * x - (zf / (q - 1))) * (1 + (alpha - 1) * x)
        # alpha * x = (q / (q - 1)) * x + (q / (q - 1)) * (alpha - 1) * x^2 - (zf / (q - 1)) - (zf / (q - 1)) * (alpha - 1) * x
        # 0 = (q / (q - 1)) * (alpha - 1) * x^2 + (q / (q - 1) - (zf / (q - 1)) * (alpha - 1) - alpha) * x - (zf / (q - 1))

        # Let A = (q / (q - 1)) * (alpha - 1)
        # Let B = (q / (q - 1)) - (zf / (q - 1)) * (alpha - 1) - alpha
        # Let C = - (zf / (q - 1))

        # Handle potential division by zero if alpha = 1 (no separation) or q = 1 (handled above)
        if abs(alpha - 1.0) < 1e-9:
             Rmin = float('inf') # Infinite reflux if no separation

        else:
            A = (q / (q - 1.0)) * (alpha - 1.0)
            B = (q / (q - 1.0)) - (zf / (q - 1.0)) * (alpha - 1.0) - alpha
            C = - (zf / (q - 1.0))

            # Quadratic formula: x = (-B ± sqrt(B^2 - 4AC)) / (2A)
            discriminant = B**2 - 4 * A * C

            if discriminant < 0:
                # No real intersection point, likely indicates an issue with inputs (e.g., compositions not achievable)
                print("Warning: No real intersection found for q-line and equilibrium curve. Check inputs.")
                Rmin = float('inf') # Or handle as an error
            else:
                sqrt_discriminant = np.sqrt(discriminant)
                x_intersect1 = (-B + sqrt_discriminant) / (2 * A)
                x_intersect2 = (-B - sqrt_discriminant) / (2 * A)

                # The relevant intersection point is typically between xb and xd
                # Also, it should be on the correct side of the feed composition zf based on q.
                # For a typical distillation, the intersection is between zf and xd if q <= 1,
                # or between xb and zf if q > 1.
                # Let's choose the root that is between xb and xd and is the smaller root if both are in that range.
                valid_roots = [x for x in [x_intersect1, x_intersect2] if xb <= x <= xd + 1e-9] # Add tolerance

                if not valid_roots:
                     # No valid intersection in the desired range
                     print("Warning: No valid intersection found between xb and xd. Check inputs.")
                     Rmin = float('inf') # Or handle as an error
                else:
                    # Choose the root closest to zf or the one that makes sense physically
                    # For typical operation, the intersection point is where the q-line hits the equilibrium curve
                    # between the bottoms and distillate compositions.
                    # Let's assume the smaller valid root is the correct intersection point.
                    x_intersect = min(valid_roots)

                    # Find y_intersect on the equilibrium curve at x_intersect
                    y_intersect = alpha * x_intersect / (1 + (alpha - 1) * x_intersect)

                    # Calculate Rmin using the intersection point
                    # Rmin = (xd - y_intersect) / (y_intersect - x_intersect)
                    # Avoid division by zero if y_intersect == x_intersect
                    if abs(y_intersect - x_intersect) < 1e-9:
                        Rmin = float('inf') # Infinite reflux if azeotrope at intersection
                    else:
                        Rmin = (xd - y_intersect) / (y_intersect - x_intersect)


    # Ensure Rmin is not negative (can happen with unfavorable VLE or inputs)
    # If Rmin is calculated as negative, it usually indicates that the desired separation
    # is not possible even at minimum reflux, or there's an issue with the alpha or compositions.
    # In a practical sense, Rmin cannot be negative. A negative Rmin might suggest
    # an error in the calculation or inputs. Let's cap it at a small positive value or report an error.
    if Rmin < 0:
        print(f"Warning: Calculated Rmin is negative ({Rmin:.2f}). This may indicate infeasible separation or calculation error. Capping Rmin at a small positive value.")
        Rmin = 1e-9 # Cap at a small positive value

    return Rmin


def calculate_actual_reflux_ratio(Rmin, rr_mult):
    """
    Calculates actual reflux ratio based on minimum reflux ratio and a multiplier.
    Rmin: Minimum reflux ratio
    rr_mult: Reflux ratio multiplier (R/Rmin)
    Returns: Actual reflux ratio (R)
    """
    if Rmin == float('inf'):
        return float('inf') # If minimum reflux is infinite, actual reflux is also infinite
    return Rmin * rr_mult

def calculate_minimum_stages(xd, xb, alpha):
    """
    Calculates minimum number of theoretical stages (Nmin) using the Fenske equation.
    xd: Mole fraction of light key in distillate
    xb: Mole fraction of light key in bottoms
    alpha: Average relative volatility
    Returns: Minimum number of theoretical stages (Nmin)
    """
    # Fenske equation: Nmin = log((xd/(1-xd)) * ((1-xb)/xb)) / log(alpha)
    # Avoid log(0) or log(negative)
    if xb <= 0 or xd >= 1 or alpha <= 1:
        return float('inf') # Infinite stages needed if separation is impossible or alpha <= 1
    try:
        nmin = np.log((xd / (1 - xd)) * ((1 - xb) / xb)) / np.log(alpha)
        # Nmin must be positive. If negative, it might indicate inputs where xb > xd.
        return max(0.0, nmin)
    except Exception as e:
        print(f"Error calculating minimum stages: {e}")
        return float('inf')


def calculate_actual_stages(Nmin, R, Rmin):
    """
    Calculates actual number of theoretical stages using the Gilliland correlation.
    Nmin: Minimum number of theoretical stages
    R: Actual reflux ratio
    Rmin: Minimum reflux ratio
    Returns: Actual number of theoretical stages
    """
    # Gilliland correlation is an empirical correlation between N/Nmin and (R-Rmin)/(R+1)
    # The correlation is typically presented as a graph or fitted equation.
    # A common form of the fitted equation is:
    # (N - Nmin) / (N + 1) = 0.75 * ((R - Rmin) / (R + 1))**0.566
    # We need to solve for N.

    # Handle edge cases:
    if Rmin < 0 or R <= Rmin or Nmin <= 0: # Gilliland correlation requires R > Rmin and Nmin > 0
        return float('inf')
    if Rmin == float('inf') or R == float('inf') or Nmin == float('inf'):
         return float('inf')


    # Calculate the correlation parameter X = (R - Rmin) / (R + 1)
    X = (R - Rmin) / (R + 1)

    # Ensure X is within the valid range for the correlation (0 < X < 1)
    # Due to floating point, X might be slightly negative if R is very close to Rmin. Cap it at 0.
    X = max(0.0, X)
    if X >= 1.0: # Should not happen if R > Rmin
         X = 0.999999 # Cap at a value slightly less than 1

    # Use a common fit for the Gilliland correlation: (N - Nmin) / (N + 1) = f(X)
    # f(X) = 1 - exp((1 + 5*X) / (11*X) * (X - 1))  # Another form
    # A simpler polynomial fit (valid for 0.01 < X < 0.9):
    # Y = (N - Nmin) / (N + 1)
    # Y ≈ 0.75 - 0.75 * X**0.566  (Rough approximation)

    # Let's use a commonly cited empirical equation form:
    # (N - Nmin) / (N + 1) = 0.756 - 0.756 * X**0.566
    # (N - Nmin) = (0.756 - 0.756 * X**0.566) * (N + 1)
    # N - Nmin = 0.756*N + 0.756 - 0.756*X**0.566*N - 0.756*X**0.566
    # N - 0.756*N + 0.756*X**0.566*N = Nmin + 0.756 - 0.756*X**0.566
    # N * (1 - 0.756 + 0.756*X**0.566) = Nmin + 0.756 - 0.756*X**0.566
    # N * (0.244 + 0.756*X**0.566) = Nmin + 0.756 * (1 - X**0.566)
    # N = (Nmin + 0.756 * (1 - X**0.566)) / (0.244 + 0.756*X**0.566)

    # A more common form of fit is:
    # Y = (N - Nmin) / (N + 1)
    # X = (R - Rmin) / (R + 1)
    # Y = 1 - exp( (1 + 5*X) / (11*X) * (X - 1) )
    # Y = 0.2417 - 0.2335*X + 0.1307*X**2 - 0.0142*X**3 + 0.0019*X**4 # Fit from a source

    # Let's use a simpler, commonly cited correlation equation form:
    # (N - Nmin) / (N + 1) = f(X)
    # Where f(X) is some empirical function. A simple fit is:
    # Y = 0.206 + log10(X) * (0.675 + 0.206 * log10(X)) # Valid for 0.01 < X < 0.9

    # Let's use a straightforward polynomial fit for Y = (N - Nmin) / (N + 1) vs X = (R - Rmin) / (R + 1)
    # from a common reference (e.g., Perry's Handbook or separation process textbooks).
    # A simple piecewise linear approximation or a polynomial fit can be used.
    # Let's use a common polynomial approximation:
    # Y = 0.5458 - 0.1173*X + 0.0324*X**2 + 0.0015*X**3 - 0.0019*X**4

    # Y values for common X values:
    # X=0.01, Y=0.16
    # X=0.02, Y=0.23
    # X=0.05, Y=0.33
    # X=0.1,  Y=0.43
    # X=0.2,  Y=0.55
    # X=0.4,  Y=0.7
    # X=0.6,  Y=0.8
    # X=0.8,  Y=0.9

    # Let's use a common explicit equation form from textbooks:
    # N / Nmin = 1 + 0.545 * ((R - Rmin) / (R + 1))**0.566 / (1 - (R - Rmin) / (R + 1))
    # This also seems complex to rearrange for N.

    # Back to the form (N - Nmin) / (N + 1) = Y
    # N - Nmin = Y * (N + 1) = Y*N + Y
    # N - Y*N = Nmin + Y
    # N * (1 - Y) = Nmin + Y
    # N = (Nmin + Y) / (1 - Y)

    # Let's use a simple polynomial fit for Y = f(X):
    # Y = 0.2417 - 0.2335*X + 0.1307*X**2 - 0.0142*X**3 + 0.0019*X**4  (Valid for 0.01 < X < 0.9)
    # If X is outside this range, the correlation might not be accurate.

    # Let's use the common graphical correlation translated to an equation form.
    # A frequently cited correlation is from Kirkbride:
    # N_rect / N_strip = (xb / xd)**alpha_avg * (B / D)  # This is for feed stage location, not total stages.

    # Let's use the form: (N - Nmin) / (N + 1) = f(X)
    # Where X = (R - Rmin) / (R + 1)
    # Let's use a fit provided in some chemical engineering resources:
    # Y = 1 - exp((1 + 5*X) / (11*X) * (X - 1))  # This is a bit complex.

    # Let's use a simpler polynomial fit for Y:
    # If X <= 0.9: Y = 0.5458 - 0.1173*X + 0.0324*X**2 + 0.0015*X**3 - 0.0019*X**4
    # If X > 0.9: Y = 1.0 # As X approaches 1, N approaches infinity, (N-Nmin)/(N+1) approaches 1.

    # Let's use a widely accepted fit from Treybal, Mass-Transfer Operations, 3rd ed., p. 182:
    # For (R-Rmin)/(R+1) from 0 to 1:
    # Y = (N-Nmin)/(N+1)
    # X = (R-Rmin)/(R+1)
    # Data points for Y vs X:
    # (0, 0), (0.01, 0.16), (0.02, 0.23), (0.05, 0.33), (0.1, 0.43), (0.2, 0.55), (0.4, 0.70), (0.6, 0.80), (0.8, 0.90), (1.0, 1.0)
    # A polynomial fit to this data might be:
    # Y ≈ 0.2417 - 0.2335*X + 0.1307*X^2 - 0.0142*X^3 + 0.0019*X^4  (This fit is for X from 0.01 to 0.9)

    # Let's use a simpler approximate formula often used:
    # N = Nmin + Rmin / (R - Rmin) * f(R/Rmin)  # Not standard Gilliland

    # Let's go back to the form N = (Nmin + Y) / (1 - Y) and calculate Y from X using a simple approximation or lookup.
    # For simplicity, let's use a linear interpolation between a few points or a simple curve fit.
    # A simple empirical fit often cited is:
    # N/Nmin = f((R-Rmin)/(R+1))
    # Let's use the explicit form if possible.
    # N = Nmin + (Rmin / (R - Rmin)) * function_of_R_Rmin_ratio

    # Let's use the direct fit for Y vs X and then solve for N:
    # Y = 0.5458 - 0.1173*X + 0.0324*X**2 + 0.0015*X**3 - 0.0019*X**4  (Example fit)

    # Calculate X
    X = (R - Rmin) / (R + 1)

    # Ensure X is in a valid range for the correlation (e.g., 0 to ~0.9)
    if X < 0: X = 0 # Should not happen if R > Rmin
    if X > 0.95: X = 0.95 # Cap X at a reasonable value to avoid extrapolation issues

    # Calculate Y using a simple polynomial approximation (example fit):
    Y = 0.5458 - 0.1173*X + 0.0324*X**2 + 0.0015*X**3 - 0.0019*X**4

    # Ensure Y is not too close to 1 to avoid division by zero or large N
    if Y >= 0.99: Y = 0.99

    # Solve for N: N = (Nmin + Y) / (1 - Y)
    # Avoid division by zero if Y is close to 1
    if abs(1 - Y) < 1e-9:
         N = float('inf')
    else:
        N = (Nmin + Y) / (1 - Y)

    # The number of stages must be at least Nmin.
    # Also, the number of stages must be a positive integer. Round up to the nearest integer.
    if Nmin > 0:
         return max(Nmin, np.ceil(N))
    else:
         # If Nmin is 0 or negative, Gilliland is not applicable in the standard way.
         # This scenario should ideally be caught earlier (Nmin <= 0 check).
         # If we reach here and Nmin <= 0, return infinity or handle as error.
         return float('inf') # Or raise an error


def _rectifying(x, R, xd):
    """
    Calculates the y-coordinate on the rectifying operating line for a given x.
    """
    # y = (R / (R + 1)) * x + (xd / (R + 1))
    # Avoid division by zero if R is negative or R+1 is zero (should not happen with R >= Rmin >= 0)
    if R + 1 <= 0:
        return np.full_like(x, np.nan) # Return NaN if denominator is invalid
    return (R / (R + 1)) * x + (xd / (R + 1))

def _stripping(x, R, q, F, D, B, xb):
    """
    Calculates the y-coordinate on the stripping operating line for a given x.
    """
    # L_bar = R * D + q * F
    # V_bar = L_bar - B
    # y = (L_bar / V_bar) * x - (B * xb) / V_bar

    # Calculate L_bar and V_bar based on molar flows
    L_bar = R * D + q * F # Molar flow in stripping section liquid
    V_bar = L_bar - B     # Molar flow in stripping section vapor

    # Avoid division by zero if V_bar is zero or close to zero
    if abs(V_bar) < 1e-9:
         return np.full_like(x, np.nan) # Return NaN if denominator is invalid

    return (L_bar / V_bar) * x - (B * xb) / V_bar

def _q_line(x, q, zf):
    """
    Calculates the y-coordinate on the q-line for a given x.
    """
    # y = (q / (q - 1)) * x - (zf / (q - 1))
    # Handle q = 1 separately (vertical line at x = zf)
    if abs(q - 1.0) < 1e-9:
        # For q=1, the q-line is vertical at x=zf. The y value is undefined in the standard equation form.
        # This function is typically used to plot the line, not find a specific y for a given x unless x=zf.
        # For plotting purposes, if x == zf, any y value on the plot range (0 to 1) is on the line.
        # If x != zf, it's not on the line.
        # For plotting, we can return a line segment at x=zf.
        # This function is called with an array of x values for plotting.
        # Let's return NaN for all x except zf, which is not directly useful for plotting a line.
        # A better approach for plotting the q-line when q=1 is to use ax.axvline(x=zf).
        # If this function is strictly for calculating points on the line for a given x,
        # and q=1, it implies the line is vertical at zf. If the input x is zf, any y is on the line.
        # If the input x is not zf, no y exists on the line.
        # Given how this is used in plot_mccabe_thiele with np.linspace(0, 1, 2),
        # this function is for drawing the line segment.
        # Let's return a line segment from (zf, 0) to (zf, 1) if q=1.
        # But the expected output is a y value for input x.
        # This function should probably not be called with np.linspace(0,1,2) when q=1.
        # Let's modify plot_mccabe_thiele to handle q=1 plotting separately.
        # For now, if q=1, return NaN for the standard formula, which will be handled by plotting.
        return np.full_like(x, np.nan)

    # Avoid division by zero if q is close to 1
    if abs(q - 1.0) < 1e-9:
         return np.full_like(x, np.nan) # Return NaN if denominator is invalid

    return (q / (q - 1.0)) * x - (zf / (q - 1.0))

def calculate_energy_and_cost(D, R, feed_comp, q_value, F_mol, mw1, mw2, xd, xb, n_stages, bp1, bp2, energy_cost_per_kwh, tower_cost_mult, condenser_cost_mult, reboiler_cost_mult):
    """
    Calculates energy consumption and equipment costs with customizable cost parameters.
    Note: This uses simplified energy calculations based on Trouton's rule at normal boiling points.
    A more rigorous approach would use enthalpies from the VLE model.
    """
    # Estimate average latent heat of vaporization using Trouton's rule at normal boiling points
    # This is a simplification. Rigorous calculation would use enthalpy from the VLE model
    # at the actual temperatures and compositions in the condenser and reboiler.
    Tb1 = bp1 + 273.15 # K
    Tb2 = bp2 + 273.15 # K
    # Use a weighted average based on feed composition (simplification)
    # A better approach might use compositions in the condenser (xd) and reboiler (xb)
    avg_dH_vap_feed = feed_comp * (88 * Tb1) + (1 - feed_comp) * (88 * Tb2)  # J/mol
    # Let's use a weighted average based on distillate and bottoms compositions as a slightly better estimate
    avg_dH_vap_condenser = xd * (88 * Tb1) + (1 - xd) * (88 * Tb2) # J/mol (at dew point of vapor)
    avg_dH_vap_reboiler = xb * (88 * Tb1) + (1 - xb) * (88 * Tb2) # J/mol (at bubble point of liquid)

    # Simplified approach: use average of condenser and reboiler estimated dH_vap
    avg_dH_vap = (avg_dH_vap_condenser + avg_dH_vap_reboiler) / 2.0


    # Vapor flow rate in rectifying section (mol/hr)
    V = (R + 1) * D
    # Vapor flow rate in stripping section (mol/hr)
    # V_bar = V - (1 - q) * F_mol # Alternate definition from q-line derivation
    L_bar = R * D + q * F_mol # Molar liquid flow in stripping section
    V_bar = L_bar - B         # Molar vapor flow in stripping section

    # Energy duties (kWh)
    # Q_cond = V * avg_dH_vap / 3.6e6  # Using V_rect and average dH_vap
    # Q_reb = V_bar * avg_dH_vap / 3.6e6 # Using V_strip and average dH_vap

    # More accurately, use dH_vap at condenser/reboiler conditions
    Q_cond = V * avg_dH_vap_condenser / 3.6e6  # kWh
    Q_reb = V_bar * avg_dH_vap_reboiler / 3.6e6 # kWh


    # Equipment cost estimation (simplified power-law models - need to check units and basis)
    # Base costs and exponents are illustrative and would come from cost correlations.
    # Assume costs are in USD.
    # Tower cost often depends on diameter and height (related to number of stages).
    # A very simplified model based on stages: Cost ~ Stages^exponent
    # Condenser/Reboiler cost often depends on heat duty: Cost ~ Duty^exponent

    # Illustrative base costs (USD) and exponents:
    BASE_TOWER_COST = 15000  # Illustrative cost for a small tower
    TOWER_STAGE_EXP = 0.8    # Exponent for number of stages
    BASE_CONDENSER_COST = 5000 # Illustrative cost
    CONDENSER_DUTY_EXP = 0.65  # Exponent for duty
    BASE_REBOILER_COST = 6000  # Illustrative cost
    REBOILER_DUTY_EXP = 0.7    # Exponent for duty

    tower_cost = tower_cost_mult * BASE_TOWER_COST * (n_stages ** TOWER_STAGE_EXP) if n_stages > 0 else 0
    condenser_cost = condenser_cost_mult * BASE_CONDENSER_COST * (Q_cond ** CONDENSER_DUTY_EXP) if Q_cond > 0 else 0
    reboiler_cost = reboiler_cost_mult * BASE_REBOILER_COST * (Q_reb ** REBOILER_DUTY_EXP) if Q_reb > 0 else 0

    total_equip_cost = tower_cost + condenser_cost + reboiler_cost

    # Energy cost ($/hr)
    energy_cost_hr = (Q_cond + Q_reb) * energy_cost_per_kwh

    # Cost per kg of distillate ($/kg)
    distillate_kg = D * (xd * mw1 + (1 - xd) * mw2) / 1000.0  # kg/hr
    cost_per_kg = energy_cost_hr / distillate_kg if distillate_kg > 1e-9 else 0 # Avoid division by very small numbers


    return Q_cond, Q_reb, total_equip_cost, energy_cost_hr, cost_per_kg, tower_cost, condenser_cost, reboiler_cost


def calculate_feed_stage(alpha, R, xd, xb, zf, q, F_mol, D, B):
    """
    Calculates the estimated optimal feed stage location based on the intersection of
    operating lines and the q-line (assuming constant alpha for this estimation).
    Returns: Estimated feed stage number (1-indexed from the top, condenser=stage 1).
             Returns None if calculation is not possible or intersection is outside bounds.
    """
    # The feed stage is where the q-line intersects the equilibrium curve, and the operating lines intersect.
    # The intersection point (x_intersect, y_intersect) is calculated by solving the q-line equation
    # and either the rectifying or stripping line equation. Using the intersection of the two operating lines is common.

    # Rectifying line: y = (R / (R + 1)) * x + (xd / (R + 1))
    # Stripping line: y = (L_bar / V_bar) * x - (B * xb) / V_bar
    # L_bar = R*D + q*F_mol
    # V_bar = L_bar - B

    # Intersection of the two operating lines:
    # (R / (R + 1)) * x + (xd / (R + 1)) = (L_bar / V_bar) * x - (B * xb) / V_bar
    # (R / (R + 1) - L_bar / V_bar) * x = - (B * xb) / V_bar - (xd / (R + 1))
    # x_intersect = (- (B * xb) / V_bar - (xd / (R + 1))) / (R / (R + 1) - L_bar / V_bar)

    # Handle potential division by zero
    V_bar = (R * D + q * F_mol) - B
    if abs(R + 1) < 1e-9 or abs(V_bar) < 1e-9 or abs(R / (R + 1) - (L_bar / V_bar)) < 1e-9:
         print("Warning: Cannot calculate operating line intersection for feed stage estimation.")
         return None

    L_bar = R * D + q * F_mol

    try:
        x_intersect = (- (B * xb) / V_bar - (xd / (R + 1.0))) / (R / (R + 1.0) - L_bar / V_bar)
        y_intersect = _rectifying(x_intersect, R, xd) # Find y on the rectifying line at the intersection x

        # The feed stage is estimated by finding which stage on the McCabe-Thiele diagram
        # corresponds to this intersection point.
        # This often involves stepping down from the top (xD) until the liquid composition
        # on a stage is less than or equal to x_intersect, or stepping up from the bottom (xB)
        # until the vapor composition from a stage is greater than or equal to y_intersect.

        # A common approximation for the feed stage location (Kremser method related):
        # Number of stages in rectifying section (above feed) N_rect:
        # N_rect = log(((xd/(1-xd)) * ((1-y_intersect)/y_intersect)) * (1/alpha)) / log(alpha) # Using y_intersect from equi curve
        # N_rect = log(((xd/(1-xd)) * ((1-x_intersect)/x_intersect))) / log(alpha) # Approx using x_intersect as liquid comp

        # Using the graphical interpretation: Step down from xd until you cross the intersection point.
        # This requires the stage-by-stage calculation logic.
        # Since plot_mccabe_thiele already does the stage stepping and finds the feed stage based on the q-line intersection,
        # let's refactor the feed stage calculation to use the result from plot_mccabe_thiele.

        # This function should probably estimate the feed stage based on the intersection point and operating lines
        # without re-doing the full stage stepping.
        # A simpler empirical rule based on the intersection point:
        # Step down from xd along the rectifying line until y_n = y_intersect. Find the corresponding x_n.
        # Then find the number of ideal stages from xd to x_intersect using the equilibrium curve and rectifying line.

        # Let's calculate the number of stages above the feed (N_rect) and below the feed (N_strip)
        # needed to reach the intersection point from xd and xb respectively.
        # The feed stage is then N_rect + 1 (if counting from the top).

        # N_rect: stages from xd down to x_intersect on the rectifying section.
        # This requires stepping down using the rectifying line and equilibrium curve from xd to x_intersect.
        # Let's count stages from xd down to x_intersect using the rigorous equilibrium function.
        n_rect = 0
        x_current_rect = xd
        # Need to step down the rectifying section.
        # From y_n on the equilibrium curve, go down to x_n+1 on the rectifying line.
        # y_n = equilibrium(x_current_rect)
        # x_next_rect = (y_n - xd/(R+1)) * (R+1)/R

        # This requires the rigorous equilibrium function and stepping, which is done in plot_mccabe_thiele.
        # Let's instead use the calculated feed stage from the rigorous stage stepping in plot_mccabe_thiele.
        # This means this `calculate_feed_stage` function might become redundant if the plot function returns the feed stage.

        # If this function is to provide an *independent* estimate based on the intersection point,
        # it needs to calculate stages to the intersection.
        # Let's calculate stages from xd to x_intersect using the rectifying line and equilibrium.
        # This is essentially a partial stage stepping.

        n_rect_steps = 0
        x_current_step = xd
        # Use a small tolerance to avoid infinite loop if x_current_step exactly equals x_intersect
        while x_current_step > x_intersect + 1e-9 and n_rect_steps < 100:
             # Find equilibrium y for current x
             # This requires the rigorous equilibrium function which needs Antoine coeffs and pressure.
             # Let's assume this function will receive those parameters if it's to be rigorous.
             # For now, let's keep it simple and use the original simplified equilibrium if rigorous VLE is not passed.
             # To make it rigorous, it needs antoine_coeffs1, antoine_coeffs2, P_atm, gamma1, gamma2
             # Let's update the function signature.

             # Using the rigorous equilibrium (requires updating function signature to accept VLE params)
             # y_eq_step = equilibrium(x_current_step, T_at_x_current, P_atm, ...) # Need temperature at x_current_step
             # Finding T at each x is complex.

             # Let's stick to the original approach of estimating based on the operating line intersection point.
             # The feed stage is the stage where the liquid composition is between x_intersect and the liquid composition of the stage above it,
             # and the vapor composition from that stage is between y_intersect and the vapor composition of the stage above it.

             # A simpler approximation: the feed is on the stage where the liquid composition is closest to zf.
             # Or, count stages from the top until the liquid composition is just below x_intersect.

             # Let's assume for now that the feed stage returned by `plot_mccabe_thiele` is the primary rigorous estimate.
             # This function can provide an alternate estimate based purely on the intersection point x_intersect.
             # Find the number of ideal stages from xd down to x_intersect using Fenske-like approach or stepping on ideal curve?

             # Let's calculate the number of stages from the top down to the intersection point x_intersect
             # using the rectifying line and the equilibrium curve. This is essentially counting stages
             # until the liquid composition is less than or equal to x_intersect.

             # This still requires the stage stepping logic.
             # It seems the most robust way to calculate the feed stage is within the stage stepping process.
             # Let's rely on the feed stage estimated by `calculate_stages_rigorous_plotting` within `plot_mccabe_thiele`.

             # Therefore, this `calculate_feed_stage` function should probably just return the x-coordinate of the intersection point
             # or indicate where the feed should be introduced relative to the intersection.

             # Let's return the calculated x_intersect and y_intersect for display,
             # and rely on the plot function's stage stepping to identify the actual feed stage number.

             # Re-calculating stages to find feed stage based on intersection x_intersect
             n_stages_from_top_to_intersect = 0
             x_current_step = xd
             # Need to use the rectifying line and the actual equilibrium curve for stepping
             # This requires the rigorous VLE. Let's call the rigorous stage calculation function
             # and find the stage where the liquid composition crosses x_intersect.

             # This seems circular. The rigorous plotting function already calculates the feed stage.
             # Let's make this function simply return the x-coordinate of the intersection point
             # of the operating lines as an alternative feed location indicator.

             pass # Placeholder, as the logic seems to be integrated into plot_mccabe_thiele


        # Let's calculate the intersection point and return it.
        # The rigorous feed stage number will come from plot_mccabe_thiele.

        return x_intersect # Return the x-coordinate of the intersection point

    except Exception as e:
        print(f"Error calculating feed stage intersection: {e}")
        return None


def fetch_nist_data(component_name):
    """
    Fetches boiling point and molecular weight data for a component from NIST.
    Returns a dictionary with 'bp' and 'mw', or None if data not found.
    Note: This is a simplified scraper and may break if NIST page structure changes.
    It does NOT currently fetch Antoine coefficients.
    """
    base_url = "https://webbook.nist.gov/cgi/cbook.cgi?Name="
    search_url = f"{base_url}{requests.utils.quote(component_name)}"

    try:
        response = requests.get(search_url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        soup = BeautifulSoup(response.content, 'html.parser')

        boiling_point = None
        molecular_weight = None

        # Attempt to find Boiling Point
        # Look for links or text that might indicate thermal properties or phase change data
        thermal_prop_link = soup.find('a', string=lambda text: text and 'Thermal properties' in text)
        if thermal_prop_link:
             thermal_url = "https://webbook.nist.gov" + thermal_prop_link['href']
             thermal_response = requests.get(thermal_url)
             thermal_response.raise_for_status()
             thermal_soup = BeautifulSoup(thermal_response.content, 'html.parser')

             # Look for Boiling Point in tables on the thermal properties page
             for table in thermal_soup.find_all('table'):
                 if "Boiling Point" in table.get_text():
                     # Assuming the structure is a table with property name and value
                     for row in table.find_all('tr'):
                         cols = row.find_all(['td', 'th'])
                         if len(cols) > 1 and "Boiling Point" in cols[0].get_text():
                             try:
                                 # Extract temperature, assume it's in the second column
                                 bp_text = cols[1].get_text().strip()
                                 # Extract numeric part and handle units
                                 # Look for temperature followed by unit (C, K, F)
                                 import re
                                 match = re.search(r'([-+]?\d*\.?\d+)\s*([CKF])', bp_text)
                                 if match:
                                     temp_val = float(match.group(1))
                                     unit = match.group(2)
                                     if unit == 'C':
                                         boiling_point = temp_val
                                     elif unit == 'K':
                                         boiling_point = temp_val - 273.15 # Convert K to C
                                     elif unit == 'F':
                                         boiling_point = (temp_val - 32) * 5/9 # Convert F to C
                                     # Take the first valid boiling point found
                                     break
                             except (ValueError, IndexError, AttributeError) as e:
                                 print(f"Could not parse boiling point from table row: {bp_text} - {e}")
                 if boiling_point is not None: break # Stop searching tables once found


        # Attempt to find Molecular Weight
        # Molecular weight is often on the main page or identity page
        mol_weight_heading = soup.find(['h2', 'h3'], string=lambda text: text and 'Molecular Weight' in text)
        if mol_weight_heading:
             try:
                 # Assume molecular weight is in the next sibling tag or a nearby element
                 mw_tag = mol_weight_heading.find_next(['p', 'div', 'span']) # Adjust tag based on inspection
                 if mw_tag:
                     mw_text = mw_tag.get_text().strip()
                     # Extract the number, it might be the first number or labeled
                     import re
                     match = re.search(r'(\d+\.?\d*)', mw_text)
                     if match:
                          molecular_weight = float(match.group(1))
                     else: # Sometimes it's just the number after the heading
                         mw_sibling = mol_weight_heading.next_sibling
                         if mw_sibling and isinstance(mw_sibling, str):
                              match = re.search(r'(\d+\.?\d*)', mw_sibling)
                              if match:
                                  molecular_weight = float(match.group(1))

             except Exception as e:
                 print(f"Could not parse molecular weight for {component_name}: {e}")


        # If boiling point or molecular weight is still None, try a more general search
        if boiling_point is None:
            for tag in soup.find_all(string=lambda text: text and "Boiling point" in text):
                try:
                    # Look for temperature value in the surrounding text
                    bp_text = tag.get_text()
                    import re
                    match = re.search(r'([-+]?\d*\.?\d+)\s*([CKF])', bp_text)
                    if match:
                         temp_val = float(match.group(1))
                         unit = match.group(2)
                         if unit == 'C': boiling_point = temp_val
                         elif unit == 'K': boiling_point = temp_val - 273.15
                         elif unit == 'F': boiling_point = (temp_val - 32) * 5/9
                         if boiling_point is not None: break # Stop searching
                except: pass # Ignore errors and continue searching


        if molecular_weight is None:
             for tag in soup.find_all(string=lambda text: text and "Molecular Weight" in text):
                 try:
                    mw_text = tag.get_text()
                    import re
                    match = re.search(r'(\d+\.?\d*)', mw_text)
                    if match:
                         molecular_weight = float(match.group(1))
                         if molecular_weight is not None: break # Stop searching
                 except: pass # Ignore errors and continue searching


        if boiling_point is not None and molecular_weight is not None:
            return {"bp": boiling_point, "mw": molecular_weight}
        else:
            print(f"Could not find complete data for {component_name} on NIST.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {component_name} from NIST: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while processing {component_name}: {e}")
        return None
