import sys
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imageio
from pathlib import Path
import os
import glob
import math
import PySimpleGUI as sG

sG.theme('DarkGrey2')   # ustawienie koloru okna

col1 = [[sG.Text("Kąt początkowy pierwszej masy:")],
        [sG.Text("Kąt początkowy drugiej masy:")],
        [sG.Text("Masa pierwsza:")],
        [sG.Text("Masa druga:")],
        [sG.Text("Ramię pierwsze:")],
        [sG.Text("Ramię drugie:")],
        [sG.Text("Przyspieszenie ziemskie:")],
        [sG.Text("Klatkaż animacji:")],
        [sG.Text("Czas animacji:")],
        [sG.Text("Kolor masy pierwszej:")],
        [sG.Text("Kolor masy drugiej:")],
        [sG.Text("Rysowanie śladu masy pierwszej:")],
        [sG.Text("Rysowanie śladu masy drugiej:")]]


col2 = [[sG.Input(default_text=0)],  # value 0
        [sG.Input(default_text=0)],  # value 1
        [sG.Input(default_text=0.5)],  # value 2
        [sG.Input(default_text=0.5)],  # value 3
        [sG.Input(default_text=1)],  # value 4
        [sG.Input(default_text=1)],  # value 5
        [sG.Input(default_text=9.81)],  # value 6
        [sG.Input(default_text=25)],  # value 7
        [sG.Input(default_text=5)],  # value 8
        [sG.Combo(["Czerwony", "Zielony", "Niebieski", "Żółty                                     "], default_value="Zielony", key="kolor1")],  # kolor1 value9
        [sG.Combo(["Czerwony", "Zielony", "Niebieski", "Żółty                                     "], default_value="Czerwony", key="kolor2")],  # kolor2 value10
        [sG.Checkbox("Tak", key="slad1", default=True)],  # slad1 value11
        [sG.Checkbox("Tak", key="slad2", default=True)]]  # slad2 value12


col3 = [[sG.Text("stopni")],
        [sG.Text("stopni")],
        [sG.Text("kg")],
        [sG.Text("kg")],
        [sG.Text("m")],
        [sG.Text("m")],
        [sG.Text("m/s^2")],
        [sG.Text("FPS")],
        [sG.Text("s")],
        [sG.Text("")],
        [sG.Text("")],
        [sG.Text("")],
        [sG.Text("")]]

layout = [[sG.Column(col1), sG.Column(col2), sG.Column(col3)],
          [sG.Button("   Symulacja   ")],
          [sG.Output(size=(82, 5))]]

window = sG.Window("Symulator wahadła podwójnego", layout, element_justification="center")  # tworzenie okna


def wahadlo():

    try:
        alfa_start1 = int(values[0])
        alfa_start2 = int(values[1])

        # L - długości ramienia wahadła (m)
        # m - masa obciążenia (kg)
        l1 = float(values[4])
        l2 = float(values[5])
        m1 = float(values[2])
        m2 = float(values[3])

        # Przyspieszenie grawitacyjne (m/s^2)
        g = float(values[6])

        # Klatkaż animacji
        fps = int(values[7])

        # tmax - czas animacji (s)
        # dt - odstępy czasowe (s)
        tmax = int(values[8])
        dt = 0.01

        # s decyduje czy rysować ślady masy m
        s1 = values["slad1"]
        s2 = values["slad2"]

        # Promień rysowanych kulek uzaleznione od mas i kolory mas 1 i 2
        r1 = float(values[2]) ** 0.33/15
        r2 = float(values[3]) ** 0.33/15

        if values["kolor1"] == "Czerwony":
            kolor1 = "r"
        elif values["kolor1"] == "Zielony":
            kolor1 = "g"
        elif values["kolor1"] == "Niebieski":
            kolor1 = "b"
        elif values["kolor1"] == "Żółty                                     ":
            kolor1 = "y"

        if values["kolor2"] == "Czerwony":
            kolor2 = "r"
        elif values["kolor2"] == "Zielony":
            kolor2 = "g"
        elif values["kolor2"] == "Niebieski":
            kolor2 = "b"
        elif values["kolor2"] == "Żółty                                     ":
            kolor2 = "y"


        # Rysuje ślad masy m dla ostatnich trail_secs sekund
        trail_secs = 1
        try:
            current_directory = os.getcwd()
            final_directory = os.path.join(current_directory, r'frames')
            if not os.path.exists(final_directory):
                os.makedirs(final_directory)
        except OSError:
            print("Nie udało się stworzyć folderu")


        files = glob.glob('frames/*.png')  # Usuwanie poprzednich zdjęć w folderze frames
        for f in files:
            os.remove(f)


        def deriv(y, t, l1, l2, m1, m2):
            # Zwraca pochodne y = theta1, z1, theta2, z2

            theta1, z1, theta2, z2 = y

            c, s = np.cos(theta1 - theta2), np.sin(theta1 - theta2)  # definicja cosinusa i sinusa różnicy theta1 - theta2

            theta1dot = z1
            z1dot = (m2 * g * np.sin(theta2) * c - m2 * s * (l1 * z1 ** 2 * c + l2 * z2 ** 2) -
                     (m1 + m2) * g * np.sin(theta1)) / l1 / (m1 + m2 * s ** 2)

            theta2dot = z2
            z2dot = ((m1 + m2) * (l1 * z1 ** 2 * s - g * np.sin(theta2) + g * np.sin(theta1) * c) +
                     m2 * l2 * z2 ** 2 * s * c) / l2 / (m1 + m2 * s ** 2)

            return theta1dot, z1dot, theta2dot, z2dot


        def calc_E(y):  # Zwraca całkowitą energię układu

            th1, th1d, th2, th2d = y.T
            V = -(m1 + m2) * l1 * g * np.cos(th1) - m2 * l2 * g * np.cos(th2)
            T = 0.5 * m1 * (l1 * th1d) ** 2 + 0.5 * m2 * ((l1 * th1d) ** 2 + (l2 * th2d) ** 2 +
                                                          2 * l1 * l2 * th1d * th2d * np.cos(th1 - th2))
            return T + V


        t = np.arange(0, tmax + dt, dt)
        # Początkowe warunki: kąty startowe alfa zdefiniowane przez użytkownika i pochodne = 0
        y0 = np.array([math.radians(alfa_start1) + np.pi / 2, 0, math.radians(alfa_start2) + np.pi / 2, 0])

        # Numeryczne całkowanie równań ruchu
        y = odeint(deriv, y0, t, args=(l1, l2, m1, m2))

        # Sprawdza czy energia zgadza się ze stanem faktycznym
        EDRIFT = 0.05
        # Całkowita energia początkowa
        E = calc_E(y0)
        if np.max(np.sum(np.abs(calc_E(y) - E))) > EDRIFT:
            sys.exit("Nie udało się poprawnie wyznaczyć energii układu".format(EDRIFT))

        # z i theta w funkcji czasu
        theta1, theta2 = y[:, 0], y[:, 2]

        # Zamiana na współrzędne kartezjańskie położeń obu mas
        x1 = l1 * np.sin(theta1)
        y1 = -l1 * np.cos(theta1)
        x2 = x1 + l2 * np.sin(theta2)
        y2 = y1 - l2 * np.cos(theta2)

        # Maksymalna ilość punktów rysowanego śladu
        max_trail = int(trail_secs / dt)


        def make_plot(i):
            # Renderuje i zapiuje klatkę chwilowego położenia mas i prętów w chwili i
            ax.plot([0, x1[i], x2[i]], [0, y1[i], y2[i]], lw=2, c='k')
            # Kulki kolejno punktu kotwiczenia, masy 1 i masy 2
            c0 = Circle((0, 0), 0.05 / 2, fc='k', zorder=10)
            c1 = Circle((x1[i], y1[i]), r1, fc=kolor1, ec=kolor1, zorder=10)
            c2 = Circle((x2[i], y2[i]), r2, fc=kolor2, ec=kolor2, zorder=10)
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
            ax.set_xlim(-l1 - l2 - r1 - r2, l1 + l2 + r2 + r1)
            ax.set_ylim(-l1 - l2 - r1 - r2, l1 + l2 + r2 +r1)
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
    if event == sG.WIN_CLOSED:  # jeżeli uzytkownik zamyka okno
        break
    wahadlo()

