
from IPython import display
import numpy as np
import time
import matplotlib.pyplot as plt

class perceptron_plot:
    def __init__(self, X, Y, delay) -> None:
        self.X = X
        self.Y = Y
        self.delay = delay
        x1_min = np.min(X[:,0])
        x2_min = np.min(X[:,1])
        x1_max = np.max(X[:,0])
        x2_max = np.max(X[:,1])
        self.x1_min = x1_min - 0.75*(x1_max - x1_min)
        self.x1_max = x1_max + 0.75*(x1_max - x1_min)
        self.x2_min = x2_min - 0.75*(x2_max - x2_min)
        self.x2_max = x2_max + 0.75*(x2_max - x2_min)
        self.fig = plt.figure(figsize = (10,8))
        self.ax = self.fig.subplots()
        self.ax.set_xlim(self.x1_min, self.x1_max, auto=False)
        self.ax.set_ylim(self.x2_min, self.x2_max, auto=False)

    def graficar(self, W, x0, epoch, fila) -> None:
        display.clear_output(wait =True)
        plt.cla()
        #self.ax = self.fig.subplots()

        self.ax.set_xlim(self.x1_min, self.x1_max)
        self.ax.set_ylim(self.x2_min, self.x2_max)
        plt.title( 'epoch ' + str(epoch) )
        # ploteo puntos positivos
        self.ax.plot(self.X[self.Y==1,0], self.X[self.Y==1,1], 'o', color="green", markersize=10)
        # ploteo puntos negativos
        self.ax.plot(self.X[self.Y==0,0], self.X[self.Y==0,1], 'o', color="blue", markersize=10)
      
        # Sobreploteo el punto que no coincidio
        if(fila>=0):
            self.ax.plot(self.X[fila,0], self.X[fila,1], 'o', 
                         color= ('green' if self.Y[fila]==1 else 'blue'), 
                         markersize= 12, markeredgecolor= 'red')

        #dibujo le recta
        vx2_min = -(W[0]*self.x1_min + x0)/W[1]
        vx2_max = -(W[0]*self.x1_max + x0)/W[1]

        self.ax.plot([self.x1_min, self.x1_max],
                     [vx2_min, vx2_max], 
                     linewidth = 2, 
                     color = 'red', 
                     alpha = 0.5)
        
        display.display(plt.gcf())
        #plt.cla()
        time.sleep(self.delay)
     
