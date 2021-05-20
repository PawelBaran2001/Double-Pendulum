import sys
import time
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imageio
from pathlib import Path
import os
import glob
import math
import PySimpleGUI as sg

sg.theme('DarkGrey2')   #ustawienie koloru okna

col1 = [    [sg.Text("Kąt początkowy pierwszej masy:")],
            [sg.Text("Kąt początkowy drugiej masy:")],
            [sg.Text("Masa pierwsza:")],
            [sg.Text("Masa druga:")],
            [sg.Text("Ramię pierwsze:")],
            [sg.Text("Ramię drugie:")],
            [sg.Text("Przyspieszenie ziemskie:")]]

col2 = [    [sg.Input(default_text=0)],  # value 0
            [sg.Input(default_text=0)],  # value 1
            [sg.Input(default_text=0.5)],  # value2
            [sg.Input(default_text=0.5)],  # value3
            [sg.Input(default_text=1)],  # value4
            [sg.Input(default_text=1)],  # value5
            [sg.Input(default_text=9.81)], ] # value6

col3 = [    [sg.Text("stopni")],
            [sg.Text("stopni")],
            [sg.Text("kg")],
            [sg.Text("kg")],
            [sg.Text("m")],
            [sg.Text("m")],
            [sg.Text("m/s^2")]]

layout = [[sg.Column(col1), sg.Column(col2), sg.Column(col3)],
          [sg.Button("Symulacja")],
          [sg.Output(size=(82,5))]]

window = sg.Window("Symulator wahadła podwójnego", layout, element_justification="center") #tworzenie okna

def wahadlo():

    try:
        alfa_start1 = int(values[0])
        alfa_start2 = int(values[1])

        # L - długości ramienia wahadła (m)
        # m - masa obciążenia (kg)
        L1 = float(values[4])
        L2 = float(values[5])
        m1 = float(values[2])
        m2 = float(values[3])

        # Przyspieszenie grawitacyjne (m/s^2)
        g = float(values[6])

        # Klatkaż animacji
        fps = 24

        # tmax - czas animacji (s)
        # dt - odstępy czasowe (s)
        tmax = 6
        dt = 0.01

        # s decyduje czy rysować ślady masy m
        s1 = True
        s2 = True

        # Promień rysowanego koła i kolory mas 1 i 2
        r = 0.05
        kolor1 = "b"
        kolor2 = "r"

        # Rysuje ślad masy m dla ostatnich trail_secs sekund
        trail_secs = 1

        files = glob.glob('frames/*.png')  # Usuwanie poprzednich zdjęć w folderze frames
        for f in files:
            os.remove(f)


        def deriv(y, t, L1, L2, m1, m2):
            # Zwraca pochodne y = theta1, z1, theta2, z2

            theta1, z1, theta2, z2 = y

            c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)  # definicja cosinusa i sinusa różnicy theta1 - theta2

            theta1dot = z1
            z1dot = (m2 * g * np.sin(theta2) * c - m2 * s * (L1 * z1 ** 2 * c + L2 * z2 ** 2) -
                     (m1 + m2) * g * np.sin(theta1)) / L1 / (m1 + m2 * s ** 2)

            theta2dot = z2
            z2dot = ((m1 + m2) * (L1 * z1 ** 2 * s - g * np.sin(theta2) + g * np.sin(theta1) * c) +
                     m2 * L2 * z2 ** 2 * s * c) / L2 / (m1 + m2 * s ** 2)

            return theta1dot, z1dot, theta2dot, z2dot


        def calc_E(y):  # Zwraca całkowitą energię układu

            th1, th1d, th2, th2d = y.T
            V = -(m1 + m2) * L1 * g * np.cos(th1) - m2 * L2 * g * np.cos(th2)
            T = 0.5 * m1 * (L1 * th1d) ** 2 + 0.5 * m2 * ((L1 * th1d) ** 2 + (L2 * th2d) ** 2 +
                                                          2 * L1 * L2 * th1d * th2d * np.cos(th1 - th2))
            return T + V


        t = np.arange(0, tmax + dt, dt)
        # Początkowe warunki: kąty startowe alfa zdefiniowane przez użytkownika i pochodne = 0
        y0 = np.array([math.radians(alfa_start1) + np.pi / 2, 0, math.radians(alfa_start2) + np.pi / 2, 0])

        # Numeryczne całkowanie równań ruchu
        y = odeint(deriv, y0, t, args=(L1, L2, m1, m2))

        # Sprawdza czy energia zgadza się ze stanem faktycznym
        EDRIFT = 0.05
        # Całkowita energia początkowa
        E = calc_E(y0)
        if np.max(np.sum(np.abs(calc_E(y) - E))) > EDRIFT:
            sys.exit("Nie udało się poprawnie wyznaczyć energii układu".format(EDRIFT))

        # z i theta w funkcji czasu
        theta1, theta2 = y[:, 0], y[:, 2]

        # Zamiana na współrzędne kartezjańskie położeń obu mas
        x1 = L1 * np.sin(theta1)
        y1 = -L1 * np.cos(theta1)
        x2 = x1 + L2 * np.sin(theta2)
        y2 = y1 - L2 * np.cos(theta2)

        # Maksymalna ilość punktów rysowanego śladu
        max_trail = int(trail_secs / dt)


        def make_plot(i):
            # Renderuje i zapiuje klatkę chwilowego położenia mas i prętów w chwili i
            ax.plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], lw=2, c='k')
            # Kulki kolejno punktu kotwiczenia, masy 1 i masy 2
            c0 = Circle((0, 0), r / 2, fc='k', zorder=10)
            c1 = Circle((x1[i], y1[i]), r, fc=kolor1, ec=kolor1, zorder=10)
            c2 = Circle((x2[i], y2[i]), r, fc=kolor2, ec=kolor2, zorder=10)
            ax.add_patch(c0)
            ax.add_patch(c1)
            ax.add_patch(c2)

            # Slad jest podzielony na ns segmentów i cieniowany aby znikał po pewnym czasie
            ns = 20
            s = max_trail // ns

            for j in range(ns):
                imin = i - (ns - j) * s
                if imin < 0:
                    continue
                imax = imin + s + 1
                # Ślad wygląda lepiej jeśli podniesiemy (j/ns) do kwadratu
                alpha = (j / ns) ** 2
                if s1 == True:
                    ax.plot(x1[imin:imax], y1[imin:imax], c=kolor1, solid_capstyle='butt', lw=2, alpha=alpha)
                if s2 == True:
                    ax.plot(x2[imin:imax], y2[imin:imax], c=kolor2, solid_capstyle='butt', lw=2, alpha=alpha)

            # Wyśrodkowanie obrazka i wyrównanie osi żeby były identyczne
            ax.set_xlim(-L1 - L2 - r, L1 + L2 + r)
            ax.set_ylim(-L1 - L2 - r, L1 + L2 + r)
            ax.set_aspect('equal', adjustable='box')
            plt.axis('off')
            plt.savefig('frames/_img{:04d}.png'.format(i // di), dpi=72)
            plt.cla()


        di = int(1 / fps / dt)
        fig = plt.figure(figsize=(10, 10), dpi=72)
        ax = fig.add_subplot(111)

        for i in range(0, t.size, di):
            print("Renderowanie klatek", i // di + 1, 'z', t.size // di + 1)
            make_plot(i)

        image_path = Path('frames')
        images = list(image_path.glob('*.png'))
        image_list = []
        for file_name in images:
            image_list.append(imageio.imread(file_name))

        print("Znaleziono klatek: ", len(image_list), ". Trwa tworzenie animacji ...")

        imageio.mimwrite('Animacja.gif', image_list, fps=fps)

        print("Proces zakończony sukcesem, utowrzono folder z kolejnymi klatkami oraz animację GIF")
        print("Możesz zmienić dane i wykonać symulację kolejny raz")
        return 0

    except:
        print("Podano złe dane, spróbuj jeszcze raz")
        return 0


while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:  # jeżeli uzytkownik zamyka okno
        break
    wahadlo()

