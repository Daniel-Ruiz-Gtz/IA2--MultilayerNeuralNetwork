from time import sleep
import numpy as np
import numpy
import matplotlib.pyplot as plt
import random
from matplotlib.widgets import Button
from matplotlib.widgets import Cursor
from matplotlib.backend_bases import MouseButton
from matplotlib.lines import Line2D


freqs = np.arange(2, 20, 3)

x = []
y = []
d = []
x5 = []

legend_elements = [Line2D([0], [0],  marker='.', color='w', label='Click Derecho',
                          markerfacecolor='r', markersize=10),
                   Line2D([0], [0], marker='.', color='w', label='Click Izquierdo',
                          markerfacecolor='g', markersize=10)]  
                   
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
t = np.arange(0.0, 1.0, 0.001)
s = np.sin(2*np.pi*freqs[0]*t)
plt.ylim(-2,2)
plt.xlim(-2,2)
plt.title("Red neuronal multicapa", fontsize = 20, color = "black")
ax.legend(handles=legend_elements, loc='upper left')
l, = plt.plot(x, y,marker=".", color="red",ls = "None")
f, = plt.plot(x, y,marker=".", color="green",ls = "None")

def darError(w1,w2,u,ite):
    
    return d[ite] - (1.0/(1.0+((np.e)**-((w1*x[ite]) + (w2*y[ite]) + u))))

def darResultado(w1,w2,u,ite):
    
    return (1.0/(1.0+((np.e)**-((w1*x[ite]) + (w2*y[ite]) + u))))

def darErrorY(w1,w2,u,y1,y2,ite):
    
    return d[ite] - (1.0/(1.0+((np.e)**-((w1*y1) + (w2*y2) + u))))

def darResultadoY(w1,w2,u,y1,y2):
    
    return (1.0/(1.0+((np.e)**-((w1*y1) + (w2*y2) + u))))

def hacerRedNeuronal(final,w0,w1,w2,w01,w11,w21,w02,w12,w22):
    plt.cla()
    plt.clf()
    
    plt.ylim(-2,2)
    plt.xlim(-2,2)
    
    todosPuntosX1C = []
    todosPuntosX2C = []
    todosPuntosX1P = []
    todosPuntosX2P = []
    
    cadaPuntox1 = -2
    cadaPuntox2 = -2
    
    if final:
        plt.title("Resultado de la red neuronal", fontsize = 20, color = "black")
    
    while (cadaPuntox1 < 2):
        
        cadaPuntox2 = -2
        while (cadaPuntox2 < 2):
            
            y1 = darResultadoY(w11, w21, w01,cadaPuntox1, cadaPuntox2)
            y2 = darResultadoY(w12, w22, w02,cadaPuntox1, cadaPuntox2)
        
            yn = darResultadoY(w1,w2,w0,y1,y2)
            
            
            if (yn > 0.5):
                todosPuntosX1C.append(cadaPuntox1)
                todosPuntosX2C.append(cadaPuntox2)
                
            else:
                todosPuntosX1P.append(cadaPuntox1)
                todosPuntosX2P.append(cadaPuntox2)
            
            cadaPuntox2 = cadaPuntox2 + 0.05
        
        cadaPuntox1 = cadaPuntox1 + 0.05
    
    plt.plot(todosPuntosX1C,todosPuntosX2C,marker="o",color = "#F5A9A9", ls = "None")
    plt.plot(todosPuntosX1P,todosPuntosX2P,marker="o",color = "#A9F5A9", ls = "None")
    
    iterador = 0
    
    while (iterador < len(x)):
        if (d[iterador] == 1):
            plt.plot(x[iterador],y[iterador],marker="o", color="red")
            
        else :
            plt.plot(x[iterador],y[iterador],marker="o", color="green")
            
        iterador = iterador + 1
        
    plt.show()
    
    plt.pause(0.8)


def hacerCalculo(self):
    plt.close()
    
    w0 = random.uniform(0,1)
    w1 = random.uniform(0,1)
    w2= random.uniform(0,1)
    
    w01 = random.uniform(0,1)
    w11 = random.uniform(0,1)
    w21 = random.uniform(0,1)
    
    w02 = random.uniform(0,1)
    w12 = random.uniform(0,1)
    w22 = random.uniform(0,1)
    
    y1 = 0.0
    y2 = 0.0 
    yn = 0.0
    #Y[2] = 0.0
    
    delta1 = 0.0
    delta2 = 0.0
    deltaN = 0.0
    
    teta = 0.4
    
    limite = 0.08
    
    lim = 3000 

    hacerPaso2 = True
    
    a = 0
    b = 0
    final = False

    
    while (hacerPaso2 == True):
        ite = 0
        b += 1
        auxErrores = 0
        
        while (ite < len(x)):
            
            y1 = darResultado(w11, w21, w01, ite)
            y2 = darResultado(w12, w22, w02, ite)
        
            yn = darResultadoY(w1,w2,w0,y1,y2)
            
            e = darErrorY(w1,w2,w0,y1,y2,ite)
            
            deltaN = e*(yn*(1-yn))
            
            w0 = w0 + (teta*deltaN*1)
            w1 = w1 + (teta*deltaN*y1)
            w2 = w2 + (teta*deltaN*y2)
            
            delta1 = (y1*(1-y1))*w1*deltaN
            delta2 = (y2*(1-y2))*w2*deltaN
            
            w01 = w01 + (teta*delta1*1)
            w11 = w11 + (teta*delta1*x[ite])
            w21 = w21 + (teta*delta1*y[ite])
            
            w02 = w02 + (teta*delta2*1)
            w12 = w12 + (teta*delta2*x[ite])
            w22 = w22 + (teta*delta2*y[ite])
            
            auxErrores = auxErrores + (e*e)
            
            ite += 1
            a += 1
        
        print("lim:",auxErrores/ite)
        if ((auxErrores/ite)<=limite):
            hacerPaso2 = False
        
        if (b > lim):
            limite += 0.01
            lim += 3000
            
        if (b%500 == 0):
            hacerRedNeuronal(final,w0,w1,w2,w01,w11,w21,w02,w12,w22)
    final = True
    hacerRedNeuronal(final,w0,w1,w2,w01,w11,w21,w02,w12,w22)
    


def onclick(event):
    if event.xdata != None and event.y > 37:
        print("posicion y:",event.y,"posicion x:",event.x)
        x.append(event.xdata)
        y.append(event.ydata)
        if event.button is MouseButton.LEFT:   
            print('Click Izquierdo')
            d.append(1)
        if event.button is MouseButton.RIGHT:
            print('Click Derecho')
            d.append(0)
        x5.append(1)
        x5.append(event.xdata)
        x5.append(event.ydata)
        
        auxX1 = []
        auxY1 = []
        auxX2 = []
        auxY2 = []
    
    s = 0
    while (s < len(x)):
        if (d[s] == 1):
            auxX1.append(x[s])
            auxY1.append(y[s])
        else:
            auxX2.append(x[s])
            auxY2.append(y[s])
            
        s = s + 1
        
    l.set_ydata(auxY1)
    l.set_xdata(auxX1)
    
    f.set_ydata(auxY2)
    f.set_xdata(auxX2)
    
    plt.draw()

i = plt.axes([0.80, 0.01, 0.1, 0.075])

iniciar = Button(i, 'Iniciar')
iniciar.on_clicked(hacerCalculo)

cursor = Cursor(ax, useblit=True, color='black', linewidth=1)
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()