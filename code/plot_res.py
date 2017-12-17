import matplotlib.pyplot as plt
import numpy as np
import obspy
from obspy import signal
from obspy import read
import sys

a = sys.argv[1]
event = a.split("/")[-1][:-4]
t1 = read(a)[0].normalize()
plt.figure(1)
plt.clf()
amplitude = t1.data
dt = t1.stats.delta
npts = t1.stats.npts
time = [i*dt for i in range(npts)]
plt.plot(time, amplitude, label = "Linear stacking")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.title(event)
plt.savefig(event+".pdf",dpi = 300)


a = sys.argv[2]
event = a.split("/")[-1][:-4]
t1 = read(a)[0].normalize()
plt.figure(1)
plt.clf()
amplitude = t1.data
dt = t1.stats.delta
npts = t1.stats.npts
time = [i*dt for i in range(npts)]
plt.plot(time, amplitude, label = "PWS stacking")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.title(event)
plt.savefig(event+".pdf",dpi = 300)
