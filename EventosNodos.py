"""
Created on Tue Sep 16 10:41:21 2023

@author: Ever Ortega Calderón
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from obspy import read, Stream
from datetime import date
from scipy import signal
from scipy.signal import hilbert
from obspy.signal.filter import bandpass
from obspy.core import UTCDateTime
from obspy.core.util.attribdict import AttribDict
import pandas as pd
import os

# Load station nodes and seismic events data
nodosPailas = pd.read_excel("./NodosPailasJunio23.xlsx")
catalogEventos = pd.read_csv("./Eventos.txt", delim_whitespace=True, skiprows=1, names=['Longitude', 'Latitude', 'Depth', 'Magnitude'])

# Event information
t_inicios_eventos = [UTCDateTime("2023-07-12T05:56:40"), UTCDateTime("2023-07-18T14:52:34"), UTCDateTime("2023-07-19T04:54:09"),
                     UTCDateTime("2023-07-19T16:35:28"), UTCDateTime("2023-07-19T19:58:21"), UTCDateTime("2023-07-28T09:02:54"),
                     UTCDateTime("2023-07-28T15:35:34"), UTCDateTime("2023-08-07T04:31:12"), UTCDateTime("2023-08-07T20:01:51")]

t_finales_eventos = [UTCDateTime("2023-07-12T06:03:40"), UTCDateTime("2023-07-18T14:59:34"), UTCDateTime("2023-07-19T05:01:09"),
                     UTCDateTime("2023-07-19T16:42:28"), UTCDateTime("2023-07-19T20:05:21"), UTCDateTime("2023-07-28T09:09:54"),
                     UTCDateTime("2023-07-28T15:42:34"), UTCDateTime("2023-08-07T04:38:12"), UTCDateTime("2023-08-07T20:08:51")]

dates = ["2023.07.12", "2023.07.18", "2023.07.19", "2023.07.19", "2023.07.19", "2023.07.28", "2023.07.28", "2023.08.07", "2023.08.07"]

# Iterate through each event
for i in range(0, len(dates)):
    print(i)
    numero_evento = i
    t_inicio_evento = t_inicios_eventos[i]
    t_final_evento = t_finales_eventos[i]
    date = dates[i]
    plt.figure(figsize=(18, 12))
    nodos = []
    
    # Create a list of nodes
    for i in range(1, 21):
        if i < 10:
            nodos.append("PA0" + str(i))
        else:
            nodos.append("PA" + str(i))

    # Function to remove instrument response
    def removerResp(st):
        zeros = [14164 + 0.0j, -7162 + 0.0j, 0 + 0.0j, 0 + 0.0j]
        poles = [-1720.4 + 0j, -1.2 + 0.9j, -1.2 - 0.9j]
        response = {'poles': poles, 'zeros': zeros, 'gain': 3355.4428, 'sensitivity': 76.7}
        evento = st
        evento.detrend('demean')
        evento.detrend('linear')
        evento.simulate(paz_remove=response)  # Assuming data in m/s
        evento.data *= 100
        return evento

    # Function to plot seismic data for a specific station and event
    def graficarNodo(nombreEstacion, date):
        st = read("../" + nombreEstacion + "/" + "*" + date + ".00.00.00.000.Z.miniseed")
        for k, tr in enumerate(st):
            component = tr.stats.channel
            data = tr.data
            time = np.arange(0, len(data)) * 0.01
            plt.plot(time, data)
        plt.show()

    # Function to plot seismic data for a specific event
    def graficarUnEvento(nombre):
        st = read("./" + nombre)
        for k, tr in enumerate(st):
            component = tr.stats.channel
            data = tr.data
            time = np.arange(0, len(data)) * 0.01
            plt.plot(time, data)
        plt.show()

    # Function to add SAC file information to seismic data
    def agregar_Informacion_SAC(nombreArchivo, numero_evento, dataframeNodos, dataframeEventos):
        st = read("./Evento" + str(numero_evento) + "/" + nombreArchivo)
        # Locate the station
        indice_inicio = nombreArchivo.find("_PA")

        # Check if "_PA" is found in the string
        if indice_inicio != -1:
            # Add 3 to skip "_PA" and take the next characters until the next underscore "_"
            indice_fin = indice_inicio + 5
            # Extract the substring "PA01"
            station = nombreArchivo[indice_inicio + 1:indice_fin]

        for i in range(0, 20):
            if station == str(dataframeNodos.iloc[i, 1]):
                latitudEstacion = dataframeNodos.iloc[i, 2]
                longitudEstacion = dataframeNodos.iloc[i, 3]
            else:
                pass

        for tr in st:
            tr.stats.network = 'OV'
            tr.stats.sac = AttribDict()
            tr.stats.sac.stla = latitudEstacion
            tr.stats.sac.stlo = longitudEstacion
            tr.stats.sac.evla = dataframeEventos.iloc[numero_evento, 1]
            tr.stats.sac.evlo = dataframeEventos.iloc[numero_evento, 0]
            tr.stats.sac.evdp = dataframeEventos.iloc[numero_evento, 2]
            tr.stats.sac.mag = dataframeEventos.iloc[numero_evento, 3]

            # Change channel names
            if tr.stats.channel == "EPZ":
                tr.stats.channel = "HHZ"
            elif tr.stats.channel == "EPE":
                tr.stats.channel = "HHE"
            elif tr.stats.channel == "EPN":
                tr.stats.channel = "HHN"

        return st

    # Iterate through each station
    for i, station in enumerate(nodos):
        try:
            st = read("../" + station + "/" + "*" + date + "*", starttime=t_inicio_evento, endtime=t_final_evento)
            for k, tr in enumerate(st):
                componente = tr.stats.channel
                evento_sin_resp = removerResp(tr)
                evento_sin_resp.write("./Evento" + str(numero_evento) + "/Evento" + str(numero_evento) + "_" + station + "_" + componente[-1] + ".sac", format="SAC")
              
        except:
            pass

# List all files in the event directory
nombres_de_archivos = os.listdir("./Evento" + str(numero_evento))

# Iterate through each file in the event directory
for nombre_archivo in nombres_de_archivos:
    # Add SAC file information to seismic data
    st = agregar_Informacion_SAC(str(nombre_archivo), numero_evento, nodosPailas, catalogEventos)

    # Write the updated seismic data to a new file in the event directory
    st.write("./Evento" + str(numero_evento) + "/" + str(nombre_archivo))
    

