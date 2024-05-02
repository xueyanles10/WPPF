import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Read data using read_csv, separated by spaces
file_path = 'si_whole.dat'
xrd_data = pd.read_csv(file_path, sep=r'\s+', comment='#', header=None, names=['2Theta', 'Intensity'])

x = xrd_data['2Theta']
y = xrd_data['Intensity']

# Adjust peak finding parameters
threshold_intensity = 1500
distance_min = 50
prominence_threshold = 0.10
width_range = (0.1, 10)

# Find diffraction peak positions
peaks, _ = find_peaks(y, height=threshold_intensity, distance=distance_min, prominence=prominence_threshold, width=width_range)

# Define the pseudo Voigt function
def PVf(x, x0, A, w, n):
    return (n * (1 / (1 + ((x - x0) / (w / 2)) ** 2)) + (1 - n) * np.exp(-0.693157 * ((x - x0) / (w / 2)) ** 2)) * A

from scipy.optimize import least_squares

# Define residual function for least squares fitting
def residual(params, x, y):
    return PVf(x, *params) - y

# Initialize an array to store the sum of fitted peaks
sum_of_peaks = np.zeros_like(x)
print(sum_of_peaks)

# Fit pseudo Voigt function to each peak and accumulate their contributions
for peak_idx in peaks:
    x_peak = x.values[peak_idx]
    y_peak = y.values[peak_idx]

    # Initial guess for curve fitting parameters
    p0 = (x_peak, y_peak.max(), 0.1, 0.5)

    # Perform least squares fitting
    result = least_squares(residual, p0, args=(x, y))

    # Extract fitted parameters
    fitted_params = result.x
    print(fitted_params)

    # Calculate contribution of the fitted peak
    peak_contribution = PVf(x, *fitted_params)

    # Add contribution to the sum
    sum_of_peaks += peak_contribution

# Plot original data and sum of fitted peaks
plt.plot(x, y, 'g-', label='Original Data')
plt.plot(x, sum_of_peaks, 'r--', label='Sum of Fitted Peaks')
plt.legend()
plt.show()
