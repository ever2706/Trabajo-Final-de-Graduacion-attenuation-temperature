
"""
Created on Tue Oct 10 08:07:14 2023

@author: Ever Ortega Calderón
"""
 # importing packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from obspy import read, Stream
from datetime import date
from scipy import signal
from scipy.signal import hilbert
from obspy.signal.filter import bandpass
from obspy.core import UTCDateTime
import matplotlib.dates as mdates
from multitaper import MTSpec, MTCross
import multitaper.utils as mutils
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
import os
import math
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import csv
from scipy.optimize import curve_fit
#leer los nodos, cada componente en un panel


#Fourier transform for the signal and noise, three panels for signal, one per chanel and three for noise


fig, axes = plt.subplots(3, 2, figsize=(25, 25), sharex=True)
#se adapta la funcion, pero en lugar de leer las tres componentes de un nodo, lee un nodo a la vez

#Function processing_train create a figure with the fourier transform of the signal and the noise of it
#function read one node at time
#t1: Added time at the beginning of the file time to locate the time when the signal starts.
#t2: Time before the signal from which it will be considered as noise.
#d: Width of the time window for analyzing the signal and for the noise.
def processing_train(node_id, starttime, names,t1, d, t2, color,paleta,dic_amplitudesmax,dic_bandas_amplitudesmax,dic_distancias):
    com=["HHE", "HHN", "HHZ"]
    # Information necessary for removing the instrument response.
    zeros = [14164 + 0.0j, -7162 + 0.0j, 0 + 0.0j, 0 + 0.0j]
    poles = [-1720.4 + 0j, -1.2 + 0.9j, -1.2 - 0.9j]
    response={'poles': poles, 'zeros': zeros, 'gain': 3355.4428, 'sensitivity':76.7}
    #read the data for the desired sensor
    st = read("./"+"data/"+"*"+node_id+"*")
    #We select the channel, and based on which channel it is, the corresponding panel is chosen.
    for i, tr in enumerate(st): 
        #The distance is calculated, and a color is assigned.
        lat_crater = 10.831485
        lon_crater = -85.336310
        lat2=tr.stats.sac.stla
        lon2=tr.stats.sac.stlo
        distancia=haversine_distance(lat_crater, lon_crater, lat2, lon2)
        dic_distancias[node_id]=distancia
        indice = int((distancia / 25) * len(paleta))
        color_seleccionado=color[indice]
        # Panel 1 is assigned for E, 2 for N, and 3 for Z
        if tr.stats.channel=="HHZ":
            num_panel=2
        elif tr.stats.channel=="HHE":
            num_panel=0
        elif tr.stats.channel=="HHN":
            num_panel=1
        #Remove  mean, trend and instrument response. 
        tr.detrend('demean')
        tr.detrend('linear')
        tr.simulate(paz_remove=response)
        #tr.data *=100
        tr.filter('highpass', freq=0.1, corners=2, zerophase=True)
        time_vec = tr.times(type='utcdatetime')
        n1 = starttime - (t2 + d) 
        n2 = starttime - t2        
        noise_window_idx=np.where(np.logical_and(time_vec>=n1, time_vec <= n2))
        timenoise = time_vec[noise_window_idx] 
        datanoise = tr.data[noise_window_idx]
        # Selecting a signal window.
        d1 = starttime + t1  
        d2 = starttime + (t1 + d)
        data_window_idx = np.where(np.logical_and(time_vec>=d1, time_vec <= d2))
        timedata = time_vec[data_window_idx]
        datadata = tr.data[data_window_idx]
        nw = len(timenoise)*tr.stats.delta
        # Calculating the FFT of the noise.
        Py1 = MTSpec(datanoise, nw=nw, kspec=7, dt=tr.stats.delta)
        freq1, spec1 = Py1.rspec()
        Py2 = MTSpec(datadata, nw=nw, kspec=7, dt=tr.stats.delta)
        freq2, spec2 = Py2.rspec()
        # Adding the overall maximum amplitudes.
        dic_amplitudesmax[node_id+tr.stats.channel[-1]]=np.max(spec2)
        # Adding the maximum amplitudes per frequency bands.
        for lim_banda in range (5,int(np.max(freq2)),5):
            indices_valores_menores_a_lim_banda = [indice for indice, valor in enumerate(freq2) if valor< lim_banda]
            amplitudes_seleccionadas = [spec2[indice] for indice in indices_valores_menores_a_lim_banda]
            dic_bandas_amplitudesmax[lim_banda][node_id+tr.stats.channel[-1]]=np.max(amplitudes_seleccionadas)
        # Plotting the curves.
        line3,= axes[num_panel,1].loglog(freq2, spec2, color=color_seleccionado, label=f'{names}')
        line1, = axes[num_panel,0].loglog(freq1, spec1, color=color_seleccionado, ls='--',  alpha=1.0)
        axes[num_panel,0].set_ylabel("Amplitude (m/s)", fontsize=22)
        axes[num_panel,1].set_ylabel("Amplitude (m/s)", fontsize=22)
        axes[num_panel,0].text(0.5, 0.1, com[num_panel], fontsize=20, color='black', ha='center', va='center' , transform=axes[num_panel, 0].transAxes)
        axes[num_panel,1].text(0.5, 0.1, com[num_panel], fontsize=20, color='black', ha='center', va='center' , transform=axes[num_panel, 1].transAxes)
        if num_panel==2:
           axes[num_panel,0].set_xlabel("Frequency (Hz)", fontsize=22)
           axes[num_panel,0].set_xlim(0.6, 10**1.8)
           axes[num_panel,1].set_xlabel("Frequency (Hz)", fontsize=22)
           axes[num_panel,1].set_xlim(0.6, 10**1.8)
           axes[num_panel,1].legend(fontsize=20)
        # Adjusting the size of the axes in the subplots.
        for i in range(3):  
            axes[i, 0].tick_params(axis='both', labelsize=22)  # Tamaño de las etiquetas y números en el eje X e Y
            axes[i, 1].tick_params(axis='both', labelsize=22)  # Tamaño de las etiquetas y números en el eje X e Y
        # Titles for each column of subplots.
        axes[0, 0].set_title("Noise",fontsize=20)
        axes[0, 1].set_title("Fourier Transform",fontsize=20)
    
channel = ["E","H","Z"]
names=set()
nombres_de_archivos = os.listdir("./data")
for nombreArchivo in nombres_de_archivos:
    # Locating the station.
    indice_inicio = nombreArchivo.find("_PA")
    # Checking if "_PA" is found in the string.
    if indice_inicio != -1:
        # Adding 3 to skip "_PA" and taking the next characters until the next underscore "_".
        indice_fin = indice_inicio+5
       # Extracting the substring "PA01".
        station = nombreArchivo[indice_inicio + 1:indice_fin]
    names.add(station)

names = list(names)

# Function that calculates the distance between the crater and each station based on their latitudes and longitudes using the Haversine formula.
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371 # Earth's radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance

# Defining the range of values in the color palette.
paleta = np.arange(0, 26)
# Defining the colors corresponding to each value in the palette.
color = plt.cm.plasma(paleta / 25)  

starttime1 = UTCDateTime("2023-07-12T05:56:40") +110
evento5_amplitudes_max={}
evento5_bandas_amplitudes_max={}
distancias={}
for lim_banda in range (5,50,5):
    evento5_bandas_amplitudes_max[lim_banda]={}
for i in range(0,len(names)):
    processing_train(names[i], starttime1, names[i], 0, 25, 2, color,paleta, evento5_amplitudes_max,evento5_bandas_amplitudes_max,distancias)


plt.tight_layout()
# Adding additional white space below the figure.
fig.subplots_adjust(bottom=0.10) 

# Adjusting the vertical position of the color bar in the whitespace.
cmap_custom = ListedColormap(color)
sm = ScalarMappable(cmap=cmap_custom, norm=plt.Normalize(0, 30))
sm.set_array([])

# Creating an axis for the color bar and placing it in the white space.
cax = fig.add_axes([0.2, 0.05, 0.6, 0.015]) 
cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
cbar.set_label('Distance (km)',fontsize=20)
# Adjusting the font size of the numbers on the color bar.
cbar.ax.tick_params(labelsize=20) 

fig.savefig('./salidas/transformada_formas.png', dpi=300)
plt.show()

# Distance vs attenuation plot.
plt.figure(figsize=(10, 6))
# Creating lists for the plot points.
x = []  # X-axis (distance values).
y = []  # Y-axis (values of event5_max_amplitudes).

# Mapping keys ending in E to one form, N to another form, Z to another form.
forma_mapping = {'E': 'o', 'N': 's', 'Z': '^'}

# Creating a dictionary to map the beginning of the identifier to a unique color.
inicio_color_mapping = {}
# Using a unique color palette from seaborn.
paleta_colores = sns.color_palette("tab20")
claves_procesadas = {}  # Dictionary to keep track of processed keys.

for clave, valor in evento5_amplitudes_max.items():
   # Getting the beginning of the identifier.
    inicio = clave[:4]
   # Checking if we have already assigned a color for this beginning.
    if inicio not in inicio_color_mapping:
        # Assigning a unique color to this beginning.
        inicio_color_mapping[inicio] = paleta_colores[len(inicio_color_mapping) % len(paleta_colores)]
    # Getting the color for this beginning.
    color = inicio_color_mapping[inicio]
    forma = forma_mapping[clave[-1]]
    if forma=='o':
        plt.scatter(distancias[inicio], valor, marker=forma, color=color, label=inicio)
    else:
        plt.scatter(distancias[inicio], valor, marker=forma, color=color)

# Setting up labels and title for the plot.
plt.xlabel('Distance (km)')
plt.ylabel('Max amplitudes (m/s)')
#plt.title('Max amplitudes per channel')
plt.legend()
# Adding text to explain the markers
plt.text(0.7, 0.8, 'Δ - HHZ', transform=plt.gcf().transFigure, fontsize=12)
plt.text(0.7, 0.75, '○ - HHE', transform=plt.gcf().transFigure, fontsize=12)
plt.text(0.7, 0.7, '■ - HHN', transform=plt.gcf().transFigure, fontsize=12, bbox={'facecolor': 'none', 'edgecolor': 'none'})


#SHow and save the figure
plt.savefig('./salidas/amplitudes_max_porcanales.png', dpi=300)
plt.show()


# Total attenuations plot, not by channel.
plt.figure(figsize=(10, 6))
# Dictionary to store the result.
result_dict = {}

# Iterating through the original dictionary.
for key, value in evento5_amplitudes_max.items():
    # Getting the beginning of the identifier.
    inicio = key[:4]
    # Squaring the value and adding it to the results dictionary.
    if inicio in result_dict:
        result_dict[inicio] += value ** 2
    else:
        result_dict[inicio] = value ** 2

# Calculating the square root of the sum of squares for each group.
for inicio in result_dict:
    result_dict[inicio] = math.sqrt(result_dict[inicio])

# Using a unique color palette from seaborn.
paleta_colores = sns.color_palette("tab20")

plt.figure(figsize=(18, 12))

# List to store distance and value values.
data_to_write = []

for i, (clave, valor) in enumerate(result_dict.items()):
    plt.scatter(distancias[clave], valor, color=paleta_colores[i], label=clave)
    distancia_agregar = distancias[clave]
    data_to_write.append([distancia_agregar, valor,clave])

# CSV file path
csv_file_path = './salidas/datos_grafico.csv'

# Writing the values to a CSV file.
with open(csv_file_path, mode='w') as file:
    file.write('Distance (km),Max amplitudes (m/s), Station\n')  # Headers.
    for row in data_to_write:
        file.write(f"{row[0]},{row[1]},{row[2]}\n")

# Performing the exponential fit.
X = np.array([distancias[clave] for clave in result_dict.keys()])
y = np.array(list(result_dict.values()))

# Define the model function.
def modelo_de_ajuste(x, a, b, c):
    return (a*(x**(-c)))* np.exp(-b * x)
  
# Code to fit the model to the data.
params, params_covariance = curve_fit(modelo_de_ajuste, X, y)

# Parameters of the fitted model.
a, b, c = params

# Generating x values for the fitted model.
x_modelo = np.linspace(min(X), max(X), len(y))
y_modelo = modelo_de_ajuste(x_modelo, a, b, c)

# Gráfico del modelo ajustado
plt.plot(x_modelo, y_modelo, color='red', linewidth=2, label='Trendline')


# Adding the equation of the fit line to the plot.
ecuacion = f"Best-fit equation: {a:.4e} * (x^({-c:.4f})) * exp({-b:.4f} * x)"
plt.text(0.75, 0.85, ecuacion, ha='right', va='center', transform=plt.gca().transAxes, fontsize=18)
r2 = r2_score(y, y_modelo)
# Adding the R² value to the plot.
r2_text = f"R²: {r2:.4f}"
plt.text(0.75, 0.75, r2_text, ha='right', va='center', transform=plt.gca().transAxes, fontsize=18)

# Setting up labels and title for the plot.
plt.xlabel('Distance (km)',fontsize=22)
plt.ylabel('Max amplitudes (m/s)',fontsize=22)
#plt.title('Max amplitudes vs distance '+f'Fórmula: y = {a} * e^{b}x\nR² = {r2:.2f}')
plt.tick_params(axis='x', labelsize=18) 
plt.tick_params(axis='y', labelsize=18) 
plt.legend(fontsize=17)

#Show and save figure
plt.savefig('./salidas/amplitudes_maximas.png', dpi=300)
plt.show()


print(f'Fórmula: y = {a} * e^{b}x\nR² = {r2:.2f}')


# Plot of frequencies by bands.
plt.figure(figsize=(10, 6))
bandas_valores_lista=set()
amplitudes_maximas_bandas_lista=[]
lista_estaciones=set()
for clave,valor in evento5_bandas_amplitudes_max.items():
    bandas_valores_lista.add(clave)
    for i,j in valor.items():
        lista_estaciones.add(i[:4])
    
for n in lista_estaciones:
    amplitudes_max_band=[]
    for i in bandas_valores_lista:
        suma_estacion_por_banda=0
        for j in evento5_bandas_amplitudes_max[i]:
            if n in j:
                suma_estacion_por_banda+=(evento5_bandas_amplitudes_max[i][j])**2
        amplitudes_max_band.append(math.sqrt(suma_estacion_por_banda))
    amplitudes_maximas_bandas_lista.append(amplitudes_max_band)


lista_estaciones=list(lista_estaciones)
bandas_valores_lista=list(bandas_valores_lista)
# Using a unique color palette from seaborn.
paleta_colores = sns.color_palette("tab20")
# Creating a scatter plot for each dataset.
for i in range(len(amplitudes_maximas_bandas_lista)):
    plt.scatter(bandas_valores_lista, amplitudes_maximas_bandas_lista[i], color=paleta_colores[i],label=lista_estaciones[i])

plt.xlabel('Frequency band value (Hz)')
plt.ylabel('Max amplitude (m/s)')
plt.legend()
#Show and save the figure
plt.savefig('./salidas/bandas_frecuencias.png', dpi=300)
plt.show()
        
# Frequency bands plot for a specific station

# Extracting the maximum amplitudes list for a specific station
amplitudes_maximas_bandas_lista_unaestacion = list(amplitudes_maximas_bandas_lista[0])

# Using a unique color palette from seaborn
color_palette = sns.color_palette("tab20")

# Creating a scatter plot for each dataset
plt.scatter(bandas_valores_lista, amplitudes_maximas_bandas_lista_unaestacion, color=color_palette[0], label=lista_estaciones[0])

# Labeling the axes
plt.xlabel('Frequency band value (Hz)')
plt.ylabel('Max amplitude (m/s)')

# Adding a legend
plt.legend()

# Displaying the scatter plot
plt.savefig('./outputs/frequency_bands_one_station.png', dpi=300)  # Saving the plot as an image file
plt.show()










