"""
Created on Tue Oct 30 12:45:11 2023

@author: Ever Ortega Calderón
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit

# Create a list to store CSV file paths
csv_files = []
folder_path = './'  # Path to your main folder containing CSV files

# Traverse through the folder and its subdirectories to find CSV files
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.csv'):
            csv_files.append(os.path.join(root, file))

# Uncomment to print the list of CSV files
# print(csv_files)

# Uncomment to remove a specific file from the list (if needed)
# csv_files.remove('./Evento20/salidas/datos_grafico.csv')

# Set up the figure size for plots
plt.figure(figsize=(18, 12))

# Initialize arrays for data points to be used in fitting
x_ajuste = np.array([])
y_ajuste = np.array([])

# Loop through each CSV file, read data, and append to arrays
for file in csv_files:
    df = pd.read_csv(file)
    x_ajuste = np.append(x_ajuste, df.iloc[:, 0].to_numpy())
    y_ajuste = np.append(y_ajuste, df.iloc[:, 1].to_numpy())
    # Uncomment if you want to plot points with colors
    # plt.scatter(df.iloc[:, 0], df.iloc[:, 1])

# SUPERGRAPH WITHOUT FITTING, ONLY SCATTER PLOT
plt.scatter(x_ajuste, y_ajuste)
plt.xlabel("Distance from the active crater (km)", fontsize=22)
plt.ylabel("Max amplitude (m/s)", fontsize=22)
plt.tick_params(axis='x', labelsize=18)
plt.tick_params(axis='y', labelsize=18)
plt.savefig("supergrafico.png", dpi=300)
# plt.show()
plt.close()

# SUPERGRAPH WITH FITTING
plt.figure(figsize=(18, 12))

# Define the fitting model function
def modelo_de_ajuste(x, a, b, c):
    return (a * (x ** (-c))) * np.exp(-b * x)

# Fit the model to the data
params, params_covariance = curve_fit(modelo_de_ajuste, x_ajuste, y_ajuste)
a, b, c = params

# Plot the data points
plt.scatter(x_ajuste, y_ajuste, label='Data')

# Generate x values for the fitted model
x_modelo = np.linspace(min(x_ajuste), max(x_ajuste), len(y_ajuste))
y_modelo = modelo_de_ajuste(x_modelo, a, b, c)

# Plot the fitted model
plt.plot(x_modelo, y_modelo, color='red', label='Trendline')

# Add the equation of the trendline to the plot
ecuacion = f"Best-fit equation: {a:.4e} * (x^({-c:.4f})) * exp({-b:.4f} * x)"
plt.text(0.95, 0.85, ecuacion, ha='right', va='center', transform=plt.gca().transAxes, fontsize=18)

# Calculate R^2 value
r2 = r2_score(y_ajuste, y_modelo)
# Add the R^2 value to the plot
r2_text = f"R²: {r2:.4f}"
plt.text(0.95, 0.75, r2_text, ha='right', va='center', transform=plt.gca().transAxes, fontsize=18)

plt.xlabel("Distance from the active crater (km)", fontsize=22)
plt.ylabel("Max amplitude (m/s)", fontsize=22)
plt.tick_params(axis='x', labelsize=18)
plt.tick_params(axis='y', labelsize=18)
plt.legend(fontsize=18)
plt.savefig("supergrafico_con_ajuste.png", dpi=300)
# plt.show()
plt.close()

# SUPERGRAPH WITHOUT OUTLIERS USING INTERQUARTILE RANGE AND FITTING
plt.figure(figsize=(18, 12))

# Calculate first and third quartiles for y_ajuste data
q1, q3 = np.percentile(y_ajuste, [25, 75])

# Calculate interquartile range (IQR)
iqr = q3 - q1

# Define the limit to identify outliers
outlier_limit = 1.5

# Filter data by removing outliers
x_filtrado = x_ajuste[(y_ajuste >= q1 - outlier_limit * iqr) & (y_ajuste <= q3 + outlier_limit * iqr)]
y_filtrado = y_ajuste[(y_ajuste >= q1 - outlier_limit * iqr) & (y_ajuste <= q3 + outlier_limit * iqr)]

# Fit the model to the filtered data
params, params_covariance = curve_fit(modelo_de_ajuste, x_filtrado, y_filtrado)
a, b, c = params

# Plot the data points with filtered values and the fitted model
plt.figure(figsize=(18, 12))
plt.scatter(x_filtrado, y_filtrado, label='Filtered data')

# Generate x values for the fitted model
x_modelo = np.linspace(min(x_filtrado), max(x_filtrado), 100)
y_modelo = modelo_de_ajuste(x_modelo, a, b, c)

# Calculate R^2
r2 = r2_score(y_filtrado, modelo_de_ajuste(x_filtrado, a, b, c))
print(f"Coeficiente de determinación R2: {r2}")

# Plot the fitted model
plt.plot(x_modelo, y_modelo, color='red', label='Trendline')

# Add the equation of the trendline to the plot
ecuacion = f"Best-fit equation: {a:.4e} * (x^({-c:.4f})) * exp({-b:.4f} * x)"
plt.text(0.95, 0.85, ecuacion, ha='right', va='center', transform=plt.gca().transAxes, fontsize=18)

# Add the R^2 value to the plot
r2_text = f"R²: {r2:.4f}"
plt.text(0.95, 0.75, r2_text, ha='right', va='center', transform=plt.gca().transAxes, fontsize=18)

plt.xlabel("Distance from the active crater (km)", fontsize=22)
plt.ylabel("Max amplitude (m/s)", fontsize=22)
plt.tick_params(axis='x', labelsize=18)
plt.tick_params(axis='y', labelsize=18)
plt.legend(fontsize=16)
plt.savefig("supergrafico_con_ajuste_sin_atipicos.png", dpi=300)
plt.show()
plt.close()

# SUPERGRAPH AVERAGE BY DISTANCE

# Create a DataFrame with the filtered data
data = {'Distancia': x_filtrado, 'Amplitud': y_filtrado}
df = pd.DataFrame(data)

# Group by distance and calculate the average amplitudes
promedio_por_distancia = df.groupby('Distancia')['Amplitud'].mean().reset_index()

# Fit the model to the averaged data
params, params_covariance = curve_fit(modelo_de_ajuste, promedio_por_distancia['Distancia'], promedio_por_distancia['Amplitud'])
a, b, c = params

# Generate x values for the fitted model
x_modelo = np.linspace(min(promedio_por_distancia['Distancia']), max(promedio_por_distancia['Distancia']), 100)
y_modelo = modelo_de_ajuste(x_modelo, a, b, c)

# Calculate R^2
r2 = r2_score(promedio_por_distancia['Amplitud'], modelo_de_ajuste(promedio_por_distancia['Distancia'], a, b, c))
print(f"Coeficiente de determinación R2 para promedio: {r2}")

# Plot with averaged data and fitted model
plt.figure(figsize=(10, 6))
plt.scatter(promedio_por_distancia['Distancia'], promedio_por_distancia['Amplitud'], label='Data')
plt.plot(x_modelo, y_modelo, color='red', label="Trendline")

# Add the equation of the trendline to the plot
ecuacion = f"Best-fit equation: {a:.4e} * (x^({-c:.4f})) * exp({-b:.4f} * x)"
plt.text(0.95, 0.80, ecuacion, ha='right', va='center', transform=plt.gca().transAxes, fontsize=14)

# Add the R^2 value to the plot
r2_text = f"R²: {r2:.4f}"
plt.text(0.95, 0.70, r2_text, ha='right', va='center', transform=plt.gca().transAxes, fontsize=14)

plt.xlabel("Distance from the active crater (km)", fontsize=12)
plt.ylabel("Average amplitude (m/s)", fontsize=12)
plt.legend(fontsize=12)
plt.savefig("supergrafico_promedio.png", dpi=300)
plt.show()
plt.close()

# SUPERGRAPH AVERAGE WITH FITTING AND ERROR BARS

# Calculate the standard deviation by distance
desviacion_estandar_por_distancia = df.groupby('Distancia')['Amplitud'].std().reset_index()


# Plot with averaged data and error bars (standard deviation)
plt.figure(figsize=(10, 6))
plt.errorbar(promedio_por_distancia['Distancia'], promedio_por_distancia['Amplitud'], yerr=desviacion_estandar_por_distancia['Amplitud'], fmt='o', color='blue', label='Error bars')
plt.plot(x_modelo, y_modelo, color='red', label="Trendline")

# Add the equation of the trendline to the plot
ecuacion = f"Best-fit equation: {a:.4e} * (x^({-c:.4f})) * exp({-b:.4f} * x)"
plt.text(0.95, 0.70, ecuacion, ha='right', va='center', transform=plt.gca().transAxes, fontsize=14)

# Add the R^2 value to the plot
r2_text = f"R²: {r2:.4f}"
plt.text(0.95, 0.60, r2_text, ha='right', va='center', transform=plt.gca().transAxes, fontsize=14)

plt.xlabel("Distance from the active crater (km)", fontsize=12)
plt.ylabel("Average amplitude (m/s)", fontsize=12)
plt.legend(fontsize=12)
# plt.grid(True)
plt.savefig("supergrafico_promedio_con_errorbars.png", dpi=300)
plt.show()
plt.close()