# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 09:41:17 2023

@author: ejcha
"""

from obspy import read
from obspy import read_inventory
from obspy.io.xseed import Parser
from obspy.signal import PPSD
from obspy.imaging.cm import pqlx
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from obspy import UTCDateTime


for i in range(1,21):
    try:
        if i<10:
            station="PA0"+str(i)
            
            st = read("./"+station+"/*Z.miniseed") 
            tr = st.select(channel="EPZ")[0]
            
            
            paz = {'gain': 3355.4428,
                   'poles': [-1720.4 + 0j, -1.2 + 0.9j, -1.2 - 0.9j],
                   'sensitivity': 76.7,
                   'zeros': [14164 + 0.0j, -7162 + 0.0j, 0 + 0.0j, 0 + 0.0j]}
            
            ppsd = PPSD(tr.stats, paz)
            ppsd.add(st) 
            #ppsd.plot(cmap=pqlx)
            ppsd.plot(cmap=pqlx, show=False)
            plt.savefig(station+"-PPSD.pdf")
            plt.close()
            ppsd.plot_spectrogram(show=False)
            plt.savefig(station+"-Spectrogram.pdf")
            plt.close()
        else:
            station="PA"+str(i)
            
            st = read("./"+station+"/*Z.miniseed") 
            tr = st.select(channel="EPZ")[0]
            
            
            paz = {'gain': 3355.4428,
                   'poles': [-1720.4 + 0j, -1.2 + 0.9j, -1.2 - 0.9j],
                   'sensitivity': 76.7,
                   'zeros': [14164 + 0.0j, -7162 + 0.0j, 0 + 0.0j, 0 + 0.0j]}
            
            ppsd = PPSD(tr.stats, paz)
            ppsd.add(st) 
            #ppsd.plot(cmap=pqlx)
            ppsd.plot(cmap=pqlx, show=False)
            plt.savefig(station+"-PPSD.pdf")
            plt.close()
            ppsd.plot_spectrogram(show=False)
            plt.savefig(station+"-Spectrogram.pdf")
            plt.close()
        
    except:
        pass