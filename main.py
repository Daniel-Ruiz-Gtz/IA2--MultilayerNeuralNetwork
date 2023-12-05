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

legend_elements = [Line2D([0], [0],  marker='.', color='w', label='Right Click',
                          markerfacecolor='r', markersize=10),
                   Line2D([0], [0], marker='.', color='w', label='Left Click',
                          markerfacecolor='g', markersize=10)]  

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
t = np.arange(0.0, 1.0, 0.001)
s = np.sin(2*np.pi*freqs[0]*t)
plt.ylim(-2,2)
plt.xlim(-2,2)
plt.title("Multilayer Neural Network", fontsize=20, color="black")
ax.legend(handles=legend_elements, loc='upper left')
l, = plt.plot(x, y, marker=".", color="red", ls="None")
f, = plt.plot(x, y, marker=".", color="green", ls="None")

def calculateError(w1, w2, u, ite):
    return d[ite] - (1.0/(1.0+((np.e)**-((w1*x[ite]) + (w2*y[ite]) + u))))

def calculateResult(w1, w2, u, ite):
    return (1.0/(1.0+((np.e)**-((w1*x[ite]) + (w2*y[ite]) + u))))

def calculateErrorY(w1, w2, u, y1, y2, ite):
    return d[ite] - (1.0/(1.0+((np.e)**-((w1*y1) + (w2*y2) + u))))

def calculateResultY(w1, w2, u, y1, y2):
    return (1.0/(1.0+((np.e)**-((w1*y1) + (w2*y2) + u))))

def buildNeuralNetwork(final, w0, w1, w2, w01, w11, w21, w02, w12, w22):
    plt.cla()
    plt.clf()

    plt.ylim(-2, 2)
    plt.xlim(-2, 2)

    allPointsX1C = []
    allPointsX2C = []
    allPointsX1P = []
    allPointsX2P = []

    eachPointX1 = -2
    eachPointX2 = -2

    if final:
        plt.title("Neural Network Result", fontsize=20, color="black")

    while (eachPointX1 < 2):

        eachPointX2 = -2
        while (eachPointX2 < 2):

            y1 = calculateResultY(w11, w21, w01, eachPointX1, eachPointX2)
            y2 = calculateResultY(w12, w22, w02, eachPointX1, eachPointX2)

            yn = calculateResultY(w1, w2, w0, y1, y2)

            if (yn > 0.5):
                allPointsX1C.append(eachPointX1)
                allPointsX2C.append(eachPointX2)

            else:
                allPointsX1P.append(eachPointX1)
                allPointsX2P.append(eachPointX2)

            eachPointX2 = eachPointX2 + 0.05

        eachPointX1 = eachPointX1 + 0.05

    plt.plot(allPointsX1C, allPointsX2C, marker="o", color="#F5A9A9", ls="None")
    plt.plot(allPointsX1P, allPointsX2P, marker="o", color="#A9F5A9", ls="None")

    iterator = 0

    while (iterator < len(x)):
        if (d[iterator] == 1):
            plt.plot(x[iterator], y[iterator], marker="o", color="red")

        else:
            plt.plot(x[iterator], y[iterator], marker="o", color="green")

        iterator = iterator + 1

    plt.show()

    plt.pause(0.8)


def performCalculation(self):
    plt.close()

    w0 = random.uniform(0, 1)
    w1 = random.uniform(0, 1)
    w2 = random.uniform(0, 1)

    w01 = random.uniform(0, 1)
    w11 = random.uniform(0, 1)
    w21 = random.uniform(0, 1)

    w02 = random.uniform(0, 1)
    w12 = random.uniform(0, 1)
    w22 = random.uniform(0, 1)

    y1 = 0.0
    y2 = 0.0
    yn = 0.0
    # Y[2] = 0.0

    delta1 = 0.0
    delta2 = 0.0
    deltaN = 0.0

    theta = 0.4

    limit = 0.08

    lim = 3000

    performStep2 = True

    a = 0
    b = 0
    final = False

    while (performStep2 == True):
        ite = 0
        b += 1
        auxErrors = 0

        while (ite < len(x)):

            y1 = calculateResult(w11, w21, w01, ite)
            y2 = calculateResult(w12, w22, w02, ite)

            yn = calculateResultY(w1, w2, w0, y1, y2)

            e = calculateErrorY(w1, w2, w0, y1, y2, ite)

            deltaN = e * (yn * (1 - yn))

            w0 = w0 + (theta * deltaN * 1)
            w1 = w1 + (theta * deltaN * y1)
            w2 = w2 + (theta * deltaN * y2)

            delta1 = (y1 * (1 - y1)) * w1 * deltaN
            delta2 = (y2 * (1 - y2)) * w2 * deltaN

            w01 = w01 + (theta * delta1 * 1)
            w11 = w11 + (theta * delta1 * x[ite])
            w21 = w21 + (theta * delta1 * y[ite])

            w02 = w02 + (theta * delta2 * 1)
            w12 = w12 + (theta * delta2 * x[ite])
            w22 = w22 + (theta * delta2 * y[ite])

            auxErrors = auxErrors + (e * e)

            ite += 1
            a += 1

        print("lim:", auxErrors/ite)
        if ((auxErrors/ite) <= limit):
            performStep2 = False

        if (b > lim):
            limit += 0.01
            lim += 3000

        if (b % 500 == 0):
            buildNeuralNetwork(final, w0, w1, w2, w01, w11, w21, w02, w12, w22)
    final = True
    buildNeuralNetwork(final, w0, w1, w2, w01, w11, w21, w02, w12, w22)


def onClick(event):
    if event.xdata is not None and event.y > 37:
        print("y position:", event.y, "x position:", event.x)
        x.append(event.xdata)
        y.append(event.ydata)
        if event.button is MouseButton.LEFT:
            print('Left Click')
            d.append(1)
        if event.button is MouseButton.RIGHT:
            print('Right Click')
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

startButton = Button(i, 'Start')
startButton.on_clicked(performCalculation)

cursor = Cursor(ax, useblit=True, color='black', linewidth=1)
cid = fig.canvas.mpl_connect('button_press_event', onClick)

plt.show()
