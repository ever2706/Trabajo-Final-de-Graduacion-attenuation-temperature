"""
Created on Tue Sep 12 11:08:34 2023

@author: Ever Ortega Calderón
"""
      
 
 
 # importing packages
import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import statsmodels.api as sm
import pygmt
from PIL import Image, ImageDraw, ImageFont 

#Nodos list
nodos=[]
for i in range (1,21):
    if i<10:
        nodos.append("PA0"+str(i))
    else:
        nodos.append("PA"+str(i))  


#temperature data path
folder_path = '../TemperaturaNodos'

file_list = glob.glob(folder_path + "/*.csv")

#colors for each station
colores = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
    '#008080', '#e6beff', '#9a6324', '#800000', '#aaffc3',
    '#808000', '#ffd8b1', '#000075', '#808080', '#f032e6'
]

#Function to find the data which match an specific station
def encontrar_coincidencia(elemento_a_buscar, lista_de_nombres):
    for nombre in lista_de_nombres:
        if elemento_a_buscar in nombre:
            return nombre
    return None

#lista_dataframes: list of data frames where each data frame corresponds to one station
lista_dataframes=[]
#here the CSV files with the temperature data are read and added to lista_dataframes
for elemento in nodos:
    coincidencia=encontrar_coincidencia(elemento,file_list)
    if coincidencia:
        dataframe = pd.read_csv(coincidencia, header=None, skiprows=3,sep=",")
        dataframe.iloc[:,1]=pd.to_datetime(dataframe.iloc[:,1])
        lista_dataframes.append(dataframe)
    else:
        pass


#here the figure with all the temperature data from all the the statios is cretead
plt.figure(figsize=(22,12))  # Ajusta el tamaño de la gráfica según tus preferencias
ax = plt.gca()

#the dataframes are plotted one by one on the same axis, with time in x axis and temperature in y axis
for indice, df in enumerate(lista_dataframes):
    tiempo = df.iloc[:, 1]
    temperatura = df.iloc[:, 2]
    if indice != 100000:
        ax.plot(tiempo, temperatura,color=colores[indice], label=nodos[indice])  # Agrega una etiqueta para cada curva

#Title, legend, axis label, etc are defined
plt.subplots_adjust(left=0.4,bottom=0.4) 
plt.xlabel('Date', fontsize=24)
plt.ylabel('Temperature (°C)', fontsize=21)
plt.title('Temperature for each station')
plt.legend(fontsize=18) 
plt.grid(True)
plt.xticks(rotation=45) 
plt.tight_layout()
plt.tick_params(axis='x', labelsize=18)  
plt.tick_params(axis='y', labelsize=17)  
# save the figure
plt.savefig("todas las temperaturas.png",dpi=300)
#plt.show()



## Statistical calculations for each station: min temperature, max temperature, standar deviation and average

minimos_por_estacion=[]
maximos_por_estacion=[]
desvi_estandar_por_estacion=[]
media_por_estacion=[]

for station in lista_dataframes:
    minimos_por_estacion.append(min(station[2]))
    maximos_por_estacion.append(max(station[2]))
    desvi_estandar_por_estacion.append(np.std((station[2])))
    media_por_estacion.append(np.mean((station[2])))

#Daily Statistical calculations for each station: min temperature, max temperature, standar deviation and average


min_station_day=[]
max_station_day=[]
desvi_estandar_station_day=[]
media_station_day=[]


for station in lista_dataframes:
    grupos_por_dia = station.groupby(pd.to_datetime(df[1]).dt.date)
    min_day={}
    max_day={}
    desvi_day={}
    media_day={}
    # Iterates through all the groups, one per day
    for fecha, grupo in grupos_por_dia:
        # 'fecha' is the date of the actual group
        # 'grupo' is a dataframe which have the data for the day analyzed
        min_day[fecha]=min(grupo[2])
        max_day[fecha]=max(grupo[2])
        desvi_day[fecha]=np.std(grupo[2])
        media_day[fecha]=np.mean(grupo[2])
    min_station_day.append(min_day)
    max_station_day.append(max_day)
    desvi_estandar_station_day.append(desvi_day)
    media_station_day.append(media_day)
    
#Function to plot the daily stats, the function receives:
#stat: data dictionary with the stat desired to plot
#stat_name: string with the name of the stadistic 
#graph_name: string with the name desired to save the figure
def GraficarStatsPerStationDay(stat,stat_name, graph_name):
    plt.figure(figsize=(23,14)) 
    ax = plt.gca()
    # Iterates through sublist of each list
    for j, station in enumerate(stat):
        x=list(station.keys())
        y=list(station.values())
        ax.plot(x, y, color=colores[j], label=nodos[j])
    plt.subplots_adjust(left=0.3,bottom=0.4) 
    plt.xlabel('Date', fontsize=22)
    plt.ylabel(stat_name, fontsize=18)
    plt.title(graph_name)
    plt.legend(fontsize=20)  
    plt.grid(True)
    plt.xticks(rotation=45)  
    plt.tight_layout()
    plt.tick_params(axis='x', labelsize=18)  
    plt.tick_params(axis='y', labelsize=14.5)  
    plt.savefig(graph_name+'.png', dpi=300)
    #plt.show()

#it is plotted every stadistic
GraficarStatsPerStationDay(min_station_day, "Minimum Temperature (°C)", "Minimum Temperature for each station per day")
GraficarStatsPerStationDay(max_station_day, "Maximmum Temperature (°C)", "Maximum Temperature for each station per day")
GraficarStatsPerStationDay(media_station_day, "Average Temperature (°C)", "Average Temperature for each station per day")

#Figure of average temperature in function of the distance from the active crater

#This function calculated the distance between tho points with the haversine formula, and receives the longitud and latitude of each point 
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371 # Earth's radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance

#path to the file with the coordenades of every station
path_coordenades = './NodosPailasJunio23.xlsx'

coordenades_station = pd.read_excel(path_coordenades)
#latitude and longitude of the Rincon de la Vieja active crater
lat_crater = 10.831485
lon_crater = -85.336310
#list from distances from the active crater to each station
distance_from_crater_to_station=[]
for i in range(0,20):
    lat=coordenades_station.iloc[i,2]
    long=coordenades_station.iloc[i,3]
    distance = haversine_distance(lat_crater, lon_crater, lat, long)
    distance_from_crater_to_station.append(distance)


#figure of average temperature per station as a function of the distance from Rincon de la Vieja´s active crater with a trendline
plt.figure(figsize=(18,12))  

X_train_distance_crater=np.reshape(np.array(distance_from_crater_to_station),((-1,1)))
y_train_distance_crater=np.reshape(np.array(media_por_estacion),(-1,1))
regression_distance_crater=LinearRegression()
regression_distance_crater.fit(X_train_distance_crater,y_train_distance_crater)
coeficientes_distance_crater = regression_distance_crater.coef_
intercepto_distance_crater = regression_distance_crater.intercept_
X_train_con_constante_distance_crater = sm.add_constant(X_train_distance_crater)
# regression lineal model with statsmodels
modelo_stats_distance_crater = sm.OLS(y_train_distance_crater, X_train_con_constante_distance_crater)
# adjust the model to the data
resultados_distance_crater = modelo_stats_distance_crater.fit()
# 95% confidence intervals  for the coefficients
intervalos_confianza_distance_crater = resultados_distance_crater.conf_int(alpha=0.05)
# plot distance vs the average temperature per station 
for j,station  in enumerate(distance_from_crater_to_station):
    plt.scatter(distance_from_crater_to_station[j],media_por_estacion[j],color=colores[j],label=nodos[j])
# plot the trendline
plt.plot(X_train_distance_crater,regression_distance_crater.predict(X_train_distance_crater),color="red")
#plot the error bars with the standar deviation
plt.errorbar(distance_from_crater_to_station,media_por_estacion,linestyle="None",yerr=desvi_estandar_por_estacion, capsize=7, ecolor="k")

# configurate details of the figure
plt.xlabel('Distance from Rincon de la Vieja´s active crater to each station (km)', fontsize=26)
plt.ylabel("Temperature (°C)", fontsize=26)
plt.title("Temperature as a function of the distance from Rincon de la Vieja´s active crater with a trendline that has a slope of " + str(np.round(coeficientes_distance_crater[0,0],4))+ " and an intercept of "+   str(np.round(intercepto_distance_crater[0],4)))
plt.legend(fontsize=18)  
plt.grid(True)
plt.tight_layout()
plt.tick_params(axis='x', labelsize=20)  
plt.tick_params(axis='y', labelsize=20)  
plt.savefig('distance_tempmedia_station.png', dpi=300)
#plt.show()

# Average temperature per station as a function of the altitude with a trendline

#list of the altitudes from every station from a file read it previously
altitud=[]
for i in range(0,20):
    alt=coordenades_station.iloc[i,4]
    altitud.append(alt)

plt.figure(figsize=(18,12))  
X_train_altitud_crater=np.reshape(np.array(altitud),((-1,1)))
y_train_altitud_crater=np.reshape(np.array(media_por_estacion),(-1,1))
regression_altitud_crater=LinearRegression()
regression_altitud_crater.fit(X_train_altitud_crater,y_train_altitud_crater)
coeficientes_altitud_crater = regression_altitud_crater.coef_
intercepto_altitud_crater = regression_altitud_crater.intercept_
X_train_con_constante_altitud_crater = sm.add_constant(X_train_altitud_crater)
# regression lineal model with statsmodels
modelo_stats_altitud_crater = sm.OLS(y_train_altitud_crater, X_train_con_constante_altitud_crater)
# Ajust the model to the data
resultados_altitud_crater = modelo_stats_altitud_crater.fit()
# 95% confidence intervals  for the coefficients
intervalos_confianza_altitud_crater = resultados_altitud_crater.conf_int(alpha=0.05)

# plot altitude vs the average temperature per station 
for j,station  in enumerate(altitud):
    plt.scatter(altitud[j],media_por_estacion[j],color=colores[j],label=nodos[j])
#plot the error bars with the standar deviation
plt.errorbar(altitud,media_por_estacion,linestyle="None",yerr=desvi_estandar_por_estacion, capsize=7, ecolor="k")
#plot the trendline
plt.plot(X_train_altitud_crater,regression_altitud_crater.predict(X_train_altitud_crater),color="red")

# configurate details of the figure
plt.xlabel('Altitude (m)', fontsize=26)
plt.ylabel("Temperature (°C)", fontsize=26)
plt.title("Temperature as a function of the altitude with a trendline that has a slope of " + str(np.round(coeficientes_altitud_crater[0,0],4))+ " and an intercept of "+   str(np.round(intercepto_altitud_crater[0],4)))
plt.legend(fontsize=18) 
plt.grid(True)
plt.tight_layout()
plt.tick_params(axis='x', labelsize=20)  
plt.tick_params(axis='y', labelsize=20)  
plt.savefig('altitud_tempmedia_station.png', dpi=300)
#plt.show()


##### distance vs average temperature figure and altitude vs average temperature figure in one same figure
fig, ax = plt.subplots(1, 2, figsize=(18, 12))
###1st figure
for j,station  in enumerate(distance_from_crater_to_station):
    ax[0].scatter(distance_from_crater_to_station[j],media_por_estacion[j],color=colores[j],label=nodos[j])
ax[0].plot(X_train_distance_crater,regression_distance_crater.predict(X_train_distance_crater),color="red")
ax[0].errorbar(distance_from_crater_to_station,media_por_estacion,linestyle="None",yerr=desvi_estandar_por_estacion, capsize=7, ecolor="k")
# configurate details of the figure
ax[0].set_xlabel('Distance from Rincon de la Vieja´s active crater to each station (km)',fontsize=18)
ax[0].set_ylabel("Temperature (°C)", fontsize=20)
ax[0].legend(fontsize=16)  
ax[0].grid(True)
ax[0].tick_params(axis='x', labelsize=18)  
ax[0].tick_params(axis='y', labelsize=18)  

###2nd
for j,station  in enumerate(altitud):
    ax[1].scatter(altitud[j],media_por_estacion[j],color=colores[j],label=nodos[j])
ax[1].errorbar(altitud,media_por_estacion,linestyle="None",yerr=desvi_estandar_por_estacion, capsize=7, ecolor="k")
ax[1].plot(X_train_altitud_crater,regression_altitud_crater.predict(X_train_altitud_crater),color="red")
# configurate details of the figure
ax[1].set_xlabel('Altitude (m)', fontsize=18)
ax[1].set_ylabel("Temperature (°C)", fontsize=20)
ax[1].legend(fontsize=16)  
ax[1].grid(True)
ax[1].tick_params(axis='x', labelsize=18)  
ax[1].tick_params(axis='y', labelsize=18)  

# Adjusts the position of the titles
ax[0].set_title("(a)", fontsize=20, y=-0.1)  # Ajusta el valor de 'y' para cambiar la ubicación vertical del título
ax[1].set_title("(b)", fontsize=20, y=-0.1)

fig.suptitle("(a) "+("Temperature as a function of the distance from Rincon de la Vieja´s active crater with a trendline that has a slope of " + str(np.round(coeficientes_distance_crater[0,0],4))+ " and an intercept of "+   str(np.round(intercepto_distance_crater[0],4)))+
  " .(b) "+("Temperature as a function of the altitude with a trendline that has a slope of " + str(np.round(coeficientes_altitud_crater[0,0],4))+ " and an intercept of "+   str(np.round(intercepto_altitud_crater[0],4))), wrap=True
)

fig.tight_layout()
fig.savefig('altitudydistancejuntos_tempmedia_station.png', dpi=300)
#fig.show()

##Map figure with temperature per station

#CSV file with the nodes locations
Nodos = pd.read_excel('./NodosPailasJunio23.xlsx')

# Create the map with PyGMT
fig = pygmt.Figure()

#Color map based on the elevation
pygmt.makecpt(
    cmap='grayC',
    series='0/3000/1', #min elevation of -8000m and max of 5000m
    continuous=True
)

#Region and reliefs (DEM)
region = [-85.45,-85.25,10.65,10.85]
projection = 'M17c'
topo_data = "@earth_relief_01s"

#Grid map
fig.grdimage(
    grid=topo_data,
    region=region,
    projection=projection,
    shading=True,
    frame=True
)

for i in range (0,len(Nodos.Longitude)):
    pen = "0.5p"
    if i<15 :
        fig.plot(
            x=Nodos['Longitude'].values[i], 
            y=Nodos['Latitude'].values[i], 
            style=str("l0.23c+t"+str(Nodos['Nodo'].values[i])), 
            #style=str("l0.1c+tA"),
            fill="dodgerblue3", 
            pen=pen
        )
    elif i==16:
        fig.plot(
            x=Nodos['Longitude'].values[i], 
            y=Nodos['Latitude'].values[i]+0.003, 
            style=str("l0.23c+t"+str(Nodos['Nodo'].values[i])), 
            #style=str("l0.1c+tA"),
            fill="dodgerblue3", 
            pen=pen
        )
    elif i==17:
        fig.plot(
            x=Nodos['Longitude'].values[i], 
            y=Nodos['Latitude'].values[i]+0.0025, 
            style=str("l0.23c+t"+str(Nodos['Nodo'].values[i])), 
            #style=str("l0.1c+tA"),
            fill="dodgerblue3", 
            pen=pen
        )
    elif i==18:
        fig.plot(
            x=Nodos['Longitude'].values[i], 
            y=Nodos['Latitude'].values[i]+0.0035, 
            style=str("l0.23c+t"+str(Nodos['Nodo'].values[i])), 
            #style=str("l0.1c+tA"),
            fill="dodgerblue3", 
            pen=pen
        )
    else:
        fig.plot(
            x=Nodos['Longitude'].values[i], 
            y=Nodos['Latitude'].values[i]+0.0023, 
            style=str("l0.23c+t"+str(Nodos['Nodo'].values[i])), 
            #style=str("l0.1c+tA"),
            fill="dodgerblue3", 
            pen=pen
        )
    
pygmt.makecpt(cmap="roma", series=[
              15, 40])

fig.colorbar(frame=["x+lTemperature", "y+l°C"])

# Save PyGMT map like an PNG image
with pygmt.config(FONT_TITLE=8):
    fig.basemap(rose="jTL+w1.3c+lO,E,S,N+o-0.1c/1c", map_scale="jBR+w5k+o0.9c/0.5c+f")


fig.savefig('mapaheat_perstation_pygmt.png', dpi=300)
#fig.show()



#########
#heat map average temperature interpolated for the all the values in a grid
#########


###Temperature value prediccition 
division_estaciones_ficticias=150
long_estaciones_ficticias=np.linspace(-85.45,-85.25,division_estaciones_ficticias)
lat_estaciones_ficticias=np.linspace(10.65,10.85,division_estaciones_ficticias) 


columnas_y_tipos = {
    'LongitudPredicTemp': float,
    'LatitudPredicTemp': float,
}

#Empty dataframe to saved temperature data
estaciones_ficticias = pd.DataFrame(columns=columnas_y_tipos.keys())

# assigns de data type to the colums 
for columna, tipo in columnas_y_tipos.items():
    estaciones_ficticias[columna] = estaciones_ficticias[columna].astype(tipo)

contador_filas=0
for i in range(0,division_estaciones_ficticias):
    for j in range(0,division_estaciones_ficticias):
        estaciones_ficticias.loc[contador_filas,'LongitudPredicTemp']=long_estaciones_ficticias[i]
        estaciones_ficticias.loc[contador_filas,'LatitudPredicTemp']=lat_estaciones_ficticias[j]
        contador_filas=contador_filas+1
  
distance_from_crater_to_station_ficticias=[]
for i in range(0,estaciones_ficticias.shape[0]):
    lat=estaciones_ficticias.iloc[i,1]
    long=estaciones_ficticias.iloc[i,0]
    distance = haversine_distance(lat_crater, lon_crater, lat, long)
    distance_from_crater_to_station_ficticias.append(distance)

temperaturas_estaciones_ficticias=regression_distance_crater.predict(np.array(distance_from_crater_to_station_ficticias).reshape(-1,1))


# Create the map with PyGMT
fig = pygmt.Figure()

#Color map based on the elevation
pygmt.makecpt(
    cmap='grayC',
    series='0/3000/1', #min elevation of -8000m and max of 5000m
    continuous=True
)

#Region and reliefs (DEM)
region = [-85.45,-85.25,10.65,10.85]
projection = 'M17c'
topo_data = "@earth_relief_01s"

#Grid map

fig.grdimage(
    grid=topo_data,
    region=region,
    projection=projection,
    shading=True,
    transparency=40,  # Ajustar la transparencia del relieve
    frame=True
)

pygmt.makecpt(cmap="roma", series=[
              15, 40])

#Plot the earthquakes by size (Mag) and depth (color)
fig.plot(
    x=estaciones_ficticias['LongitudPredicTemp'],
    y=estaciones_ficticias['LatitudPredicTemp'],
    size=(temperaturas_estaciones_ficticias.reshape(-1))*0.01,
    fill=(temperaturas_estaciones_ficticias.reshape(-1)),
    cmap=True,
    style="cc",
    pen=None, 
    transparency = 40,
)


fig.colorbar( frame=["+lTemperature", "y+l°C"])

for i in range (0,len(Nodos.Longitude)):
    pen = "0.5p"
    # plot an uppercase "A" of size 3.5c, color fill is set to "dodgerblue3"
    if i<15 :
        fig.plot(
            x=Nodos['Longitude'].values[i], 
            y=Nodos['Latitude'].values[i], 
            style=str("l0.23c+t"+str(Nodos['Nodo'].values[i])), 
            #style=str("l0.1c+tA"),
            fill="dodgerblue3", 
            pen=pen
        )
    elif i==16:
        fig.plot(
            x=Nodos['Longitude'].values[i], 
            y=Nodos['Latitude'].values[i]+0.003, 
            style=str("l0.23c+t"+str(Nodos['Nodo'].values[i])), 
            #style=str("l0.1c+tA"),
            fill="dodgerblue3", 
            pen=pen
        )
    elif i==17:
        fig.plot(
            x=Nodos['Longitude'].values[i], 
            y=Nodos['Latitude'].values[i]+0.0025, 
            style=str("l0.23c+t"+str(Nodos['Nodo'].values[i])), 
            #style=str("l0.1c+tA"),
            fill="dodgerblue3", 
            pen=pen
        )
    elif i==18:
        fig.plot(
            x=Nodos['Longitude'].values[i], 
            y=Nodos['Latitude'].values[i]+0.0035, 
            style=str("l0.23c+t"+str(Nodos['Nodo'].values[i])), 
            #style=str("l0.1c+tA"),
            fill="dodgerblue3", 
            pen=pen
        )
    else:
        fig.plot(
            x=Nodos['Longitude'].values[i], 
            y=Nodos['Latitude'].values[i]+0.0023, 
            style=str("l0.23c+t"+str(Nodos['Nodo'].values[i])), 
            #style=str("l0.1c+tA"),
            fill="dodgerblue3", 
            pen=pen
        )
# Guardar el mapa de PyGMT como una imagen PNG
with pygmt.config(FONT_TITLE=8):
    fig.basemap(rose="jTL+w1.3c+lO,E,S,N+o-0.1c/1c", map_scale="jBR+w5k+o0.9c/0.5c+f")
    
fig.savefig('Heatmapa_pygmt.png', dpi=300)
#fig.show()


# Open the heatmap image 
imagen = Image.open('Heatmapa_pygmt.png')
# Obtiene el tamaño de la imagen
ancho, alto = imagen.size
width, height = ancho,alto
background_color = (255, 255, 255, 0)

# Crea una nueva imagen con fondo transparente
image = Image.new("RGBA", (width, height), background_color)

#Draw object to draw the reference places in the image
draw = ImageDraw.Draw(image)

#reference places
textos_coords = [
    {"texto": "km", "coordenadas": (1820, 1928)},
    {"texto": "Curubandé", "coordenadas": (564, 1372)},
    {"texto": "Hacienda Santa María", "coordenadas": (1412, 706)},
    {"texto": "Vida Aventura Nature Park", "coordenadas": (1690, 1362)},
    {"texto": "Santa Anita", "coordenadas": (240,186)},
    {"texto": "Rinconcito Eco Adventure Park", "coordenadas": (1576, 1210)},
    {"texto": "DantaVentura", "coordenadas": (578,2015 )},
    {"texto": "Laguna Hilgueros", "coordenadas": (1406, 362)},
]
# load the font
font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf", 36)
# text color (RGBA)
text_color = (0, 0, 0, 255)
for item in textos_coords:
    texto = item["texto"]
    coordenadas = item["coordenadas"]
    draw.text(coordenadas, texto, fill=text_color, font=font)
# save the blank image with the reference places
image.save("imagen_con_texto.png")
# close the image
image.close()


# open the generated image
imagen = Image.open('Heatmapa_pygmt.png')

# calculate the image size
ancho, alto = imagen.size
width, height = ancho,alto
background_color = (255, 255, 255, 0)
# create another blank image 
image = Image.new("RGBA", (width, height), background_color)
#Draw object to draw the reference places in the image to set the units of the scale
draw = ImageDraw.Draw(image)

textos_coords = [
    {"texto": "km", "coordenadas": (1820, 1928)},
]

# load the font
font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf", 36)
#text color (RGBA)
text_color = (0, 0, 0, 255)
for item in textos_coords:
    texto = item["texto"]
    coordenadas = item["coordenadas"]
    draw.text(coordenadas, texto, fill=text_color, font=font)
# save the image
image.save("imagen_con_texto_km.png")

# close the image
image.close()

#Image union

# open the images to overlay, one is the map with relieve and the stations and the other is the heatmap
imagen_fondo = Image.open('mapaheat_perstation_pygmt.png').convert('RGBA')
imagen_superpuesta = Image.open('Heatmapa_pygmt.png').convert('RGBA')
# Ajust the size of the image
imagen_superpuesta = imagen_superpuesta.resize(imagen_fondo.size)
# Adjusts the transparency of the overlay image (0 = completely transparent, 255 = completely opaque)
transparencia = 210 
# blend the two images
imagen_superpuesta = Image.blend(imagen_fondo, imagen_superpuesta, alpha=transparencia/255)
# save the image
imagen_superpuesta.save('imagen_resultante.png')
# close the images
imagen_fondo.close()
imagen_superpuesta.close()


# open the images to overlay, one is the map with relieve and the stations and the other is the heatmap
imagen_fondo = Image.open('imagen_resultante.png').convert('RGBA')
imagen_superpuesta = Image.open('imagen_con_texto.png').convert('RGBA')
# Ajust the size of the image
imagen_superpuesta = imagen_superpuesta.resize(imagen_fondo.size)
# Adjusts the transparency of the overlay image (0 = completely transparent, 255 = completely opaque)
transparencia = 154 
# blend the two images
imagen_fondo.paste(imagen_superpuesta,(0,0),imagen_superpuesta)
# save the image
imagen_fondo.save('imagen_resultante2.png')
# close the image
imagen_fondo.close()
imagen_superpuesta.close()

# open the images to overlay, one is the map with relieve and the stations and the other is the heatmap
imagen_fondo = Image.open('mapaheat_perstation_pygmt.png').convert('RGBA')
imagen_superpuesta = Image.open('imagen_con_texto.png').convert('RGBA')
# Ajust the size of the image
imagen_superpuesta = imagen_superpuesta.resize(imagen_fondo.size)
# Adjusts the transparency of the overlay image (0 = completely transparent, 255 = completely opaque)
transparencia = 50
# blend the two images
imagen_fondo.paste(imagen_superpuesta,(0,0),imagen_superpuesta)
# save the image
imagen_fondo.save('mapaconkm.png')
# close the images
imagen_fondo.close()
imagen_superpuesta.close()





