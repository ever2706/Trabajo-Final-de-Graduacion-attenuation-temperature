import pandas as pd
import pygmt
import numpy as np



#CSV file with the nodes locations
Eventos = pd.read_csv('./Eventos.txt',delim_whitespace=True)
EventosVolcan=pd.read_csv('./EventosVolcan.txt',delim_whitespace=True)
Nodos = pd.read_excel('./NodosPailasJunio23.xlsx')


#####MAPA TODOS LOS EVENTOS


# Create the map with PyGMT
fig = pygmt.Figure()

#Color map based on the elevation
pygmt.makecpt(
    cmap='grayC',
    series='0/3000/1', #min elevation of -8000m and max of 5000m
    continuous=True
)

#Region and reliefs (DEM)
"""region = [-85.61,-85.25,10.65,10.95]"""
region = [-85.45,-85.25,10.65,10.87]
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


pygmt.makecpt(cmap="polar", series=[0, 5])
for m in [3,2,1]:
    s = 0.05* 3 ** m
    fig.plot(x=-85.291467, y=10.667767, style='cc', fill='grey', pen='0.20p,black', size=[s], cmap=False, label=f"Magnitude {m}+S{s}")
   

fig.text(
    x=-85.299467, y=10.657767, text="Magnitude", font="10p,Helvetica", justify="LM", angle=0
)



fig.plot(
    x=-85.305467, y= 10.663767, style="v0.3c+eA+a30", direction=([45], [0.85]), pen="2p", fill="red3"
    )
fig.text(
    x=-85.308467, y= 10.660767, text="M3", font="10p,Helvetica", justify="ML", offset="0.1c/0c"
    )

fig.plot(
    x=-85.281467, y= 10.665767, style="v0.3c+eA+a30", direction=([155], [0.7]), pen="2p", fill="red3"
    )
fig.text(
    x=-85.281467, y= 10.665767, text="M2", font="10p,Helvetica", justify="ML", offset="0.1c/0c"
    )

#lugares de referencia


fig.text(
    y= 10.719815, x=-85.410429,  text="Curubandé ", font="10p,Helvetica", justify="ML", offset="0.1c/0c"
    )

fig.text(
    y= 10.783486 ,x=-85.316716 , text="Hacienda Santa María ", font="10p,Helvetica", justify="ML", offset="0.1c/0c"
    )

fig.text(
    y= 10.718600,x=-85.299010 , text="Vida Aventura Nature Park ", font="10p,Helvetica", justify="ML", offset="0.1c/0c"
    )
fig.text(
    y=10.833550,x=-85.432730  , text="Santa Anita ", font="10p,Helvetica", justify="ML", offset="0.1c/0c"
    )

fig.text(
    y= 10.735289,x=-85.301631 , text="Rinconcito Eco Adventure Park ", font="10p,Helvetica", justify="ML", offset="0.1c/0c"
    )

fig.text(
    y=10.658395,x=-85.400361  , text="DantaVentura ", font="10p,Helvetica", justify="ML", offset="0.1c/0c"
    )

fig.text(
    y= 10.8198827,x=-85.3323572, text=" Laguna Hilgueros", font="10p,Helvetica", justify="ML", offset="0.1c/0c"
    )


fig.text(
    y= 10.8326366, x=-85.3602620 , text="Rincón de la Vieja Volcano", font="10p,Helvetica", justify="ML", offset="0.1c/0c"
    )
fig.plot(
        x=pd.concat([Eventos.Longitude,EventosVolcan.Longitude]),
        y=pd.concat([Eventos.Latitude,EventosVolcan.Latitude]),
        size=0.05 * 3 ** pd.concat([Eventos.Magnitude,EventosVolcan.Magnitude]), 
        style='cc',
        fill=pd.concat([Eventos.Depth,EventosVolcan.Depth]),
        cmap=True,
        pen='0.5p,black',
        )

fig.colorbar(frame='af+l"Depth (km)"')

fig.text(
    x=-85.2680467, y=10.659767, text="2 km", font="8p,Helvetica-Bold", justify="LM", angle=0
)
# Add a distance scale
fig.plot(
    x=[-85.2730467, -85.2546467],
    y=[10.657767, 10.657767],
    pen="1p",
    fill="black",
    label="20 km",
)






#Se agrega el nombre del nnodo al triangulo que lo representa
for i in range (0,len(Nodos.Longitude)):
    pen = "0.5p"
    # plot an uppercase "A" of size 3.5c, color fill is set to "dodgerblue3"
    fig.plot(
        x=Nodos['Longitude'].values[i], 
        y=Nodos['Latitude'].values[i], 
        style=str("l0.26c+t"+str(Nodos['Nodo'].values[i])), 
        #style=str("l0.1c+tA"),
        fill="dodgerblue3", 
        pen=pen
    )

fig.savefig("MapaTodosEventos.png",  crop=True, dpi=1200)
#fig.show()
