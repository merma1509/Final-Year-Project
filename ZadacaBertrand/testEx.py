import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'r-o')

def init():
    ax.set_xlim(0, 3*np.pi)
    ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    xdata.append(frame)
    ydata.append(np.cos(frame))
    ln.set_data(xdata, ydata)
    return ln,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 3*np.pi, 50),
                    init_func=init, blit=True)
plt.ioff()
plt.show()
