"""
Exercice 1 : l'utilisateur sélectionne une région rectangulaire de couleur uniforme ;
on produit une image où tout pixel dont la couleur diffère de celle de la zone devient noir,
les pixels de couleur similaire restent intacts.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from pcx_reader import lire_pcx


def couleur_region(img, x0, y0, x1, y1):
    """Couleur représentative de la zone : médiane par canal (robuste au bruit)."""
    zone = img[y0:y1, x0:x1].reshape(-1, 3)
    return np.median(zone, axis=0)


def filtrer_couleur(img, couleur_ref, tolerance=30.0):
    """Garde les pixels dont la distance euclidienne RGB à couleur_ref <= tolerance,
    met les autres à (0,0,0)."""
    dist = np.linalg.norm(img.astype(float) - couleur_ref.astype(float), axis=2)
    masque = dist <= tolerance
    resultat = np.zeros_like(img)
    resultat[masque] = img[masque]
    return resultat


def appliquer(img, x0, y0, x1, y1, tolerance):
    ref = couleur_region(img, x0, y0, x1, y1)
    print(f"Zone ({x0},{y0})-({x1},{y1}) -> couleur de référence RGB = {ref.astype(int)}")
    return filtrer_couleur(img, ref, tolerance)


def interactif(img, tolerance):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.imshow(img); ax1.set_title("Sélectionnez une zone uniforme (glisser la souris)")
    ax2.imshow(np.zeros_like(img)); ax2.set_title("Résultat")

    def on_select(eclick, erelease):
        x0, x1 = sorted((int(eclick.xdata), int(erelease.xdata)))
        y0, y1 = sorted((int(eclick.ydata), int(erelease.ydata)))
        if x1 - x0 < 1 or y1 - y0 < 1:
            return
        res = appliquer(img, x0, y0, x1, y1, tolerance)
        ax2.imshow(res); ax2.set_title("Résultat (tolérance = %g)" % tolerance)
        fig.canvas.draw_idle()
        plt.imsave("resultat.png", res)

    fig._sel = RectangleSelector(ax1, on_select, useblit=True, button=[1], interactive=True)
    plt.show()


if __name__ == "__main__":
    image = lire_pcx("Ex2.pcx")
    tol = 30.0
    if len(sys.argv) == 5:       # mode non interactif : x0 y0 x1 y1
        x0, y0, x1, y1 = map(int, sys.argv[1:5])
        res = appliquer(image, x0, y0, x1, y1, tol)
        plt.imsave("resultat.png", res)
    else:
        interactif(image, tol)
