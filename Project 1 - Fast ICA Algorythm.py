# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 16:47:27 2023

@author: user
"""

"""import bibliotek"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import kurtosis
import random


"""wygenerowanie ziarna (zalazka) 
   - danej wejsciowej do generatora liczb pseudolosowych"""
np.random.seed(0)


"""wektor 1000 próbek o wartociach od 0 do 200"""
ns = np.linspace(0, 500, 20000)
"""ns = [0, 0.2, ... , 199.8, 200]"""


"""transponowana macierz sygnałow wejciowych - S - sourceSignalsMatrix - 1000 x 2"""
S = np.array([np.cos(ns * 0.3), 
              (2 * signal.square(ns * 0.15))]).T

"""Wyswietlenie wykresow sygnalow""" 
S=S.T
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(S[0], color='blue')
plt.xlabel('próbki')
plt.title('Wygenerowane sygnały')
plt.ylabel('Amplituda')

plt.subplot(2, 1, 2)
plt.plot(S[1], color='orange')
plt.ylabel('Amplituda')

plt.tight_layout()
plt.show()
S=S.T

"""macierz mieszania - A - mixingMatrix - 2 x 2"""
#wartosci macierzy sa liczbami pseudolosowymi
#A = np.random.rand(2, 2)

"""wartosci macierzy sa liczbami losowymi"""
A = np.array([[1, 2], [1, -1]])

"""transponowana macierz zmieszanych sygnalow - X - mixedSignalMatrix"""
V = (S @ A).T
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(V[0], color='blue')
plt.xlabel('próbki')
plt.title('Zmieszane sygnały')
plt.ylabel('Amplituda')

plt.subplot(2, 1, 2)
plt.plot(V[1], color='orange')
plt.ylabel('Amplituda')

plt.tight_layout()
plt.show()


"""definicja funkcji liczacej kowariancje"""
def kowariancja(x):
    mean = np.mean(x, axis=1, keepdims=True)
    n = np.shape(x)[1] - 1
    m = x - mean

    return (m @ (m.T))/n

"""definicja funkcji wybielajacej"""
def wybielanie(x):
    covarianceMatrix = kowariancja(x)
    """Wyznaczenie wartosci wlasnych (wektor) i wektorow wlasnych """
    eigenvalues, eigenvectors = np.linalg.eig(covarianceMatrix)

    """obliczanie diagonalnej macierzy z wartociami wlasnymi"""
    d = np.diag(1.0 / np.sqrt(eigenvalues))

    """obliczanie macierzy wybielania"""
    M = eigenvectors @ (d @ (eigenvectors.T))

    """macierz wybielonych sygnalow"""
    Vw = M @ x

    """funkcja zwraca macierz wybielonych sygnalow i macierz wybielania""" 
    return Vw, M




"""obliczanie wartosci srednich"""
average1 = np.mean(V[:, 0], axis = 0)
average2 = np.mean(V[:, 1], axis = 0)
average = np.array([[average1], [average2]])

"""centrowanie >>> odjecie srednich"""
centeredV = V - average

"""Obliczanie macierz kowariancji dla macierzy V"""
covarianceMatrixV = kowariancja(centeredV)


"""wywolanie funkcji wybielajacej"""
Vw, M = wybielanie(centeredV)


"""Obliczanie macierz kowariancji dla macierzy Vw """
covarianceMatrixVw = np.round(np.cov(Vw))

"""Wyswietlenie macierzy kowariancji dla wybielonej macierzy Vw """
plt.figure(figsize=(10, 6))
plt.imshow(covarianceMatrixVw, cmap='winter_r')
"""Wywietlenie wartosci elementow macierzy """
for i in range(covarianceMatrixVw.shape[0]):
    for j in range(covarianceMatrixVw.shape[1]):
        plt.text(j, i, int(covarianceMatrixVw[i, j]),
                 ha="center", va="center", color="black", size=40)
plt.title("Macierz kowariancji wybielonej macierzy Vw")
plt.axis('off')
plt.show()


"""wykresy sygnalow po wybieleniu"""
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(Vw[1], color='blue')
plt.xlabel('próbki')
plt.title('Wybielone mieszanki sygnałów')
plt.ylabel('Amplituda')

plt.subplot(2, 1, 2)
plt.plot(Vw[0], color='orange')
plt.ylabel('Amplituda')

plt.tight_layout()
plt.show()


"""ALGORYTM FAST ICA"""

"""przyjecie pierwszej wartosci w10 w sposob losowy"""
w11 = random.random()
"""wyznaczenie drugiej wartosci w20, aby norma macierzy W0 == 1"""
w21 = np.sqrt(1 - w11**2)
w0 = np.array([w11, w21])
"""Sprawdzenie normy wektora W0"""
norm = np.linalg.norm(w0)
"""Stransponowanie wektora W0"""
w0 = w0.T

"""Utworzenie macierzy wybielonych mieszanin """
Z = np.array([Vw[0,:], Vw[1,:]])

"""inicjacja wektora kurtozy"""
k1 = np.zeros(shape=100)
i = 0
"""Pierwsza iteracja"""
wi = np.mean(Z * (w0.T @ Z)**3, axis=1) - 3 * w0
wi = wi / np.linalg.norm(wi)
k1[i] = kurtosis(wi.T @ Z)
i = i + 1
wynik1iter = wi.T @ Z


"""Kolejne iteracje"""
while(((abs(wi.T @ w0)) <= 0.99) or ((abs(wi.T @ w0)) >= 1.01)):
    w0 = wi
    wi = np.mean(Z * (w0.T @ Z)**3, axis=1) - 3 * w0
    wi = wi / np.linalg.norm(wi)
    k1[i] = kurtosis(wi.T @ Z)
    i = i + 1

"""nowy wykres słupkowy"""
plt.figure(figsize=(10, 6))
x = np.arange(100)
k = k1.copy()
k[k == 0] = np.nan
plt.bar(x, k, width=0.4)
plt.xticks(np.arange(8), np.arange(8))

"""wektor ortogonalny do wi"""
vi = np.array([-wi[1], wi[0]])

czyW1razyVto0 = (wi.T @ vi == 0)

"""Utworzenie macierzy W"""
W = np.array([wi.T, vi.T])

"""Rozseparowanie sygnalow"""
Y = W @ Z

"""Wyswietlanie estymowanych sygnalow """
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(Y[1], color='blue')
plt.xlabel('próbki')
plt.title('estymowane sygnały')
plt.ylabel('Amplituda')

plt.subplot(2, 1, 2)
plt.plot(Y[0], color='orange')
plt.ylabel('Amplituda')

plt.tight_layout()
plt.show()