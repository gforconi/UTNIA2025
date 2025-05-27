import numpy as np
import matplotlib.pyplot as plt

from graficos import perceptron_plot 


# Ejemplo
entrada = [ [0.7, 1.3], [2.0, 1.1], [1.0, 1.9],
            [3.0, 1.0], [1.5, 2.1], [0.5, 2] ]
# salida = [0,0,1,1,1, 0] # salucion 2
# salida = [0,0,1,0,1, 1] # salucion 1
salida = [0,0,0,1,1, 1] # no resuelve

# Paso las listas a numpy
X = np.array(entrada)
Y = np.array(salida)

#incializo los graficos
grafico = perceptron_plot(X, Y, 0.01)

# Tamano datos
X_row = X.shape[0]
X_col = X.shape[1]

# Incializo la recta azarosamente
np.random.seed(102191) #mi querida random seed para que las corridas sean reproducibles
W = np.array( np.random.uniform(-0.5, 0.5, size=X_col))
x0 = np.random.uniform(-0.5, 0.5)


# Inicializo la iteracion
epoch_limit = 500    # para terminar si no converge
learning_rate = 0.01
modificados = 1      # lo debo poner algo distinto a 0 la primera vez
epoch = 0 

while (modificados and (epoch < epoch_limit)):
    epoch += 1
    modificados = 0  #lo seteo en cero

    #recorro siempre TODA la entrada
    for fila in range(X_row):
        # calculo el estimulo suma, producto interno
        estimulo = x0*1 + W[0]*X[fila,0] + W[1]*X[fila,1]

        # funcion de activacion, a lo bruto con un if
        if(estimulo>0):
            y = 1
        else:
            y = 0

        # solo si corresponde actualizo  W y x0
        # si la salida no es la esperada se actualizan los pesos y el umbral
        # si la salida es la esperada no se hace nada
        if(y != Y[fila]):
            modificados += 1  # encontre un registro que esta mal clasificado
            # actualizo W y x0
            W[0] = W[0] + learning_rate * (Y[fila]-y) * X[fila,0]
            W[1] = W[1] + learning_rate * (Y[fila]-y) * X[fila,1]
            x0 =   x0   + learning_rate * (Y[fila]-y) * 1
            grafico.graficar(W, x0, epoch, fila) #grafico

        

grafico.graficar(W, x0, epoch, -1)
print(epoch, W, x0)