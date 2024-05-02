import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Specify the path to the .dat file
file_path = 'si_whole.dat'

# Read data using read_csv, separated by spaces
xrd_data = pd.read_csv(file_path, sep=r'\s+', comment='#', header=None, names=['2Theta', 'Intensity'])

# Extract data
x = xrd_data['2Theta']
y = xrd_data['Intensity']

# Interpolate data to get a continuous function
f = interp1d(x, y, kind='cubic')

# Define a range of x values for plotting
x_range = np.linspace(min(x), max(x), 1000)

# Calculate first derivative
dy_dx = np.gradient(f(x_range), x_range)

# Calculate second derivative
d2y_dx2 = np.gradient(dy_dx, x_range)

# Plot the original data, first derivative, and second derivative
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(x, y, label='Original Data')
plt.xlabel('2Theta')
plt.ylabel('Intensity')
plt.title('Original Data')

plt.subplot(1, 3, 2)
plt.plot(x_range, dy_dx, label='First Derivative')
plt.xlabel('2Theta')
plt.ylabel('Intensity Derivative')
plt.title('First Derivative')

plt.subplot(1, 3, 3)
plt.plot(x_range, d2y_dx2, label='Second Derivative')
plt.xlabel('2Theta')
plt.ylabel('Intensity Derivative')
plt.title('Second Derivative')

plt.tight_layout()
plt.show()
