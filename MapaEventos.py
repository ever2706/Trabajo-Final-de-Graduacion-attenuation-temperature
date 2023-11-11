"""
Created on Tue Nov 04 10:05:54 2023

@author: Ever Ortega Calderón
"""
import pandas as pd
import pygmt
import numpy as np



# CSV file with the nodes locations
Events = pd.read_csv('./Events.txt', delim_whitespace=True)
VolcanoEvents = pd.read_csv('./VolcanoEvents.txt', delim_whitespace=True)

# Map of events
region = [-87.1333333, -84.025, 9.7875, 10.908333333]
grid = pygmt.datasets.load_earth_relief(resolution="15s", region=region)

fig = pygmt.Figure()
fig.basemap(region=region, projection='M15c', frame=True)
fig.grdimage(grid=grid, projection="M15c", frame="a", cmap="broc")
fig.coast(land="grey", shorelines=["1/0.5p,black", "2/0.5p,black", "3/0.5p,black"], borders="1/1p,black")
pygmt.makecpt(cmap="polar", series=[0, 200])

# Plotting magnitudes and corresponding symbols
for m in [5, 4, 3, 2, 1]:
    s = 0.05 * 2 ** m
    fig.plot(x=-86, y=10.05, style='cc', fill='grey', pen='0.5p,black', size=[s], cmap=False, label=f"Magnitude {m}+S{s}")

fig.text(
    x=-86.15, y=9.85, text="Magnitude", font="9p,Helvetica", justify="LM", angle=0
)

# Plot symbols and labels for specific magnitudes
fig.plot(
    x=-86.22, y=9.9, style="v0.3c+eA+a30", direction=([45], [0.5]), pen="2p", fill="red3"
)
fig.text(
    x=-86.29, y=9.87, text="M5", font="6p,Helvetica", justify="ML", offset="0.1c/0c"
)

fig.plot(
    x=-86.22, y=10.0, style="v0.3c+eA+a30", direction=([20], [0.72]), pen="2p", fill="red3"
)
fig.text(
    x=-86.29, y=9.97, text="M4", font="6p,Helvetica", justify="ML", offset="0.1c/0c"
)

fig.plot(
    x=-85.82, y=9.92, style="v0.3c+eA+a30", direction=([138], [0.9]), pen="2p", fill="red3"
)
fig.text(
    x=-85.89, y=9.89, text="M3", font="6p,Helvetica", justify="ML", offset="0.1c/0c"
)

# Plot seismic events on the map with depth information and color gradient
fig.plot(
    x=Events.Longitude,
    y=Events.Latitude,
    size=0.05 * 2 ** Events.Magnitude,
    style='cc',
    fill=Events.Depth,
    cmap=True,
    pen='0.5p,black',
)

fig.colorbar(frame='af+l"Depth (km)"')

fig.text(
    x=-84.2, y=10.85, text="20 km", font="8p,Helvetica-Bold", justify="LM", angle=0
)

# Add a distance scale
fig.plot(
    x=[-84.213, -84.03],
    y=[10.8, 10.8],
    pen="1p",
    fill="black",
    label="20 km",
)

# Create an inset map with a rectangle highlighting an area
with fig.inset(position="jBL+w2.5c+o0.5c/0.2c", box="+pblack"):
    fig.coast(
        region=[-86.5, -81, 8, 12.5],
        projection="M3c",
        land="gray",
        borders=[1, 2],
        shorelines="1/thin",
        water="white",
    )
    rectangle = [[-85.534003, 11, -85.12, 11.5]]
    fig.plot(data=rectangle, style="r+s", pen="1p,red")

fig.savefig("MapaEventos.png", crop=True, dpi=1200)
#fig.show()



# Map of volcano-related events
region = [-86.1304, -85.0288, 10.30, 10.9042]
grid = pygmt.datasets.load_earth_relief(resolution="15s", region=region)

fig = pygmt.Figure()
fig.basemap(region=region, projection='M15c', frame=True)
fig.grdimage(grid=grid, projection="M15c", frame="a", cmap="broc")
fig.coast(land="grey", shorelines=["1/0.5p,black", "2/0.5p,black", "3/0.5p,black"], borders="1/1p,black" )                   
pygmt.makecpt(cmap="polar", series=[0, 10])

# Plotting magnitudes and corresponding symbols
for m in [5, 4, 3, 2, 1]:
    s = 0.05 * 2 ** m
    fig.plot(x=-86, y=10.05, style='cc', fill='grey', pen='0.5p,black', size=[s], cmap=False, label=f"Magnitude {m}+S{s}")

fig.text(
    x=-86.15, y=9.85, text="Magnitude", font="9p,Helvetica", justify="LM", angle=0
)

# Plot symbols and labels for specific magnitudes
fig.plot(
    x=-86.22, y=9.9, style="v0.3c+eA+a30", direction=([45], [0.5]), pen="2p", fill="red3"
)
fig.text(
    x=-86.29, y=9.87, text="M5", font="6p,Helvetica", justify="ML", offset="0.1c/0c"
)

fig.plot(
    x=-86.22, y=10.0, style="v0.3c+eA+a30", direction=([20], [0.72]), pen="2p", fill="red3"
)
fig.text(
    x=-86.29, y=9.97, text="M4", font="6p,Helvetica", justify="ML", offset="0.1c/0c"
)

fig.plot(
    x=-85.82, y=9.92, style="v0.3c+eA+a30", direction=([138], [0.9]), pen="2p", fill="red3"
)
fig.text(
    x=-85.89, y=9.89, text="M3", font="6p,Helvetica", justify="ML", offset="0.1c/0c"
)

# Plot volcano-related events on the map with depth information and color gradient
fig.plot(
    x=VolcanoEvents.Longitude,
    y=VolcanoEvents.Latitude,
    size=0.05 * 2 ** VolcanoEvents.Magnitude,
    style='cc',
    fill=VolcanoEvents.Depth,
    cmap=True,
    pen='0.5p,black',
)

fig.colorbar(frame='af+l"Depth (km)"')

fig.text(
    x=-85.1, y=10.87, text="5 km", font="8p,Helvetica-Bold", justify="LM", angle=0
)

# Add a distance scale
fig.plot(
    x=[-85.1, -85.054],
    y=[10.85, 10.85],
    pen="1p",
    fill="black",
    label="5 km",
)

# Create an inset map with a rectangle highlighting an area
with fig.inset(position="jBL+w2.5c+o0.5c/0.2c", box="+pblack"):
    fig.coast(
        region=[-86.5, -81, 8, 12.5],
        projection="M3c",
        land="gray",
        borders=[1, 2],
        shorelines="1/thin",
        water="white",
    )
    rectangle = [[-85.534003, 11, -85.12, 11.5]]
    fig.plot(data=rectangle, style="r+s", pen="1p,red")


fig.savefig("MapaEventosVolcan.png",  crop=True, dpi=1200)
#fig.show()






# Map of all seismic events
region = [-87.1333333, -84.025, 9.50, 10.958333333]
grid = pygmt.datasets.load_earth_relief(resolution="15s", region=region)

fig = pygmt.Figure()
fig.basemap(region=region, projection='M15c', frame=True)
fig.grdimage(grid=grid, projection="M15c", frame="a", cmap="broc")
fig.coast(land="grey", shorelines=["1/0.5p,black", "2/0.5p,black", "3/0.5p,black"], borders="1/1p,black" )                   
pygmt.makecpt(cmap="polar", series=[0, 200])

# Plotting magnitudes and corresponding symbols
for m in [5, 4, 3, 2, 1]:
    s = 0.05 * 2 ** m
    fig.plot(x=-86, y=9.7625, style='cc', fill='grey', pen='0.5p,black', size=[s], cmap=False, label=f"Magnitude {m}+S{s}")

fig.text(
    x=-86.15, y=9.5625, text="Magnitude", font="9p,Helvetica", justify="LM", angle=0
)

# Plot symbols and labels for specific magnitudes
fig.plot(
    x=-86.22, y=9.6125, style="v0.3c+eA+a30", direction=([45], [0.5]), pen="2p", fill="red3"
)
fig.text(
    x=-86.29, y=9.5825, text="M5", font="6p,Helvetica", justify="ML", offset="0.1c/0c"
)

fig.plot(
    x=-86.22, y=9.7125, style="v0.3c+eA+a30", direction=([20], [0.72]), pen="2p", fill="red3"
)
fig.text(
    x=-86.29, y=9.6825, text="M4", font="6p,Helvetica", justify="ML", offset="0.1c/0c"
)

fig.plot(
    x=-85.82, y=9.6325, style="v0.3c+eA+a30", direction=([138], [0.9]), pen="2p", fill="red3"
)
fig.text(
    x=-85.89, y=9.6025, text="M3", font="6p,Helvetica", justify="ML", offset="0.1c/0c"
)

# Reference locations
fig.text(
    y=10.635147, x=-85.440795, text="Liberia", font="6p,Helvetica", justify="ML", offset="0.1c/0c"
)
fig.text(
    y=10.058375, x=-85.419567, text="Hojancha", font="6p,Helvetica", justify="ML", offset="0.1c/0c"
)
fig.text(
    y=10.144645, x=-85.453041, text="Nicoya", font="6p,Helvetica", justify="ML", offset="0.1c/0c"
)
fig.text(
    y=10.261106, x=-85.584430, text="Santa Cruz", font="6p,Helvetica", justify="ML", offset="0.1c/0c"
)
fig.text(
    y=10.525537, x=-85.254191, text="Bagaces", font="6p,Helvetica", justify="ML", offset="0.1c/0c"
)

# Plot all seismic events on the map with depth information and color gradient
fig.plot(
    x=pd.concat([Events.Longitude, EventsVolcano.Longitude]),
    y=pd.concat([Events.Latitude, EventsVolcano.Latitude]),
    size=0.05 * 2 ** pd.concat([Events.Magnitude, EventsVolcano.Magnitude]), 
    style='cc',
    fill=pd.concat([Events.Depth, EventsVolcano.Depth]),
    cmap=True,
    pen='0.5p,black',
)

fig.colorbar(frame='af+l"Depth (km)"')

fig.text(
    x=-84.2, y=10.85, text="20 km", font="8p,Helvetica-Bold", justify="LM", angle=0
)

# Add a distance scale
fig.plot(
    x=[-84.213, -84.03],
    y=[10.8, 10.8],
    pen="1p",
    fill="black",
    label="20 km",
)
# Reference locations continued
fig.text(
    y=10.83000, x=-85.536207, text="Rincón de la Vieja Volcano", font="6p,Helvetica", justify="ML", offset="0.1c/0c"
)

fig.text(
    y=10.458349, x=-84.971256, text="Tilarán", font="6p,Helvetica", justify="ML", offset="0.1c/0c"
)

fig.text(
    y=10.428442, x=-85.093870, text="Cañas", font="6p,Helvetica", justify="ML", offset="0.1c/0c"
)

fig.text(
    y=10.086395, x=-84.730511, text="Miramar", font="6p,Helvetica", justify="ML", offset="0.1c/0c"
)

fig.text(
    y=10.468346, x=-84.642790, text="La Fortuna", font="6p,Helvetica", justify="ML", offset="0.1c/0c"
)

# Inset map
with fig.inset(position="jBL+w2.5c+o0.5c/0.2c", box="+pblack"):
    # Use a plotting function to create a figure inside the inset
    fig.coast(
        region=[-86.5, -81, 8, 12.5],
        projection="M3c",
        land="gray",
        borders=[1, 2],
        shorelines="1/thin",
        water="white",
        # dcw="CR+gorange",
        # Use dcw to selectively highlight an area
    )
    rectangle = [[-86.034003, 9.70, -84.3, 11.6]]
    fig.plot(data=rectangle, style="r+s", pen="1p,red")

# Save the figure
fig.savefig("MapaTodosEventos.png",  crop=True, dpi=1200)
#fig.show()
