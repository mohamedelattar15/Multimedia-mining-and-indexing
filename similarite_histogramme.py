# -*- coding: utf-8 -*-
"""
Similarité d'images par la méthode de l'histogramme.

Pour chaque image du dossier "Images", on calcule l'histogramme de couleurs
(en niveau de gris ou en RGB) puis on compare les histogrammes deux à deux.

Métriques de comparaison disponibles :
  - corrélation (correlation)
  - chi²        (chi_square)
  - intersection (intersection)
  - distance de Bhattacharyya (bhattacharyya)
  - distance de Hellinger    (hellinger)
  - distance euclidienne     (euclidean)
"""

import os
from itertools import combinations

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# 1) Calcul de l'histogramme
# ---------------------------------------------------------------------------
def histogramme(image_path, mode="gray", bins=256):
    """
    Calcule l'histogramme normalisé d'une image.

    Paramètres
    ----------
    image_path : str
        Chemin vers l'image.
    mode : str
        "gray" -> histogramme en niveaux de gris (1 canal).
        "rgb"  -> histogramme de couleur (3 canaux concaténés).
    bins : int
        Nombre de bins par canal.

    Retourne
    --------
    numpy.ndarray de forme (bins,) ou (bins * 3,), normalisé (somme = 1).
    """
    img = Image.open(image_path)

    if mode == "gray":
        img = img.convert("L")
        hist = np.histogram(np.asarray(img), bins=bins, range=(0, 256))[0]
    elif mode == "rgb":
        img = img.convert("RGB")
        arr = np.asarray(img)
        hist = []
        for canal in range(3):
            h = np.histogram(arr[:, :, canal], bins=bins, range=(0, 256))[0]
            hist.append(h)
        hist = np.concatenate(hist)
    else:
        raise ValueError("mode doit être 'gray' ou 'rgb'")

    # Normalisation pour que les histogrammes soient comparables
    # quelle que soit la taille de l'image.
    total = hist.sum()
    if total > 0:
        hist = hist / float(total)
    return hist


# ---------------------------------------------------------------------------
# 2) Métriques de similarité / distance
# ---------------------------------------------------------------------------
def correlation(h1, h2):
    """Corrélation : 1 = parfaitement identique, -1 = opposé."""
    h1 = h1 - h1.mean()
    h2 = h2 - h2.mean()
    num = np.sum(h1 * h2)
    den = np.sqrt(np.sum(h1 ** 2) * np.sum(h2 ** 2))
    if den == 0:
        return 0.0
    return num / den


def chi_square(h1, h2):
    """Chi² : 0 = identique, plus grand = plus différent."""
    num = (h1 - h2) ** 2
    den = h1 + h2
    # Évite la division par zéro (bins vides dans les deux images).
    mask = den > 0
    return 0.5 * np.sum(num[mask] / den[mask])


def intersection(h1, h2):
    """Intersection : 1 = identique, 0 = aucune intersection."""
    return np.sum(np.minimum(h1, h2))


def bhattacharyya(h1, h2):
    """Distance de Bhattacharyya : 0 = identique, plus grand = différent."""
    bc = np.sum(np.sqrt(h1 * h2))
    bc = min(bc, 1.0)
    return -np.log(bc) if bc > 0 else float("inf")


def hellinger(h1, h2):
    """Distance de Hellinger : 0 = identique, 1 = complètement différent."""
    return np.sqrt(1.0 - np.sum(np.sqrt(h1 * h2)))


def euclidean(h1, h2):
    """Distance euclidienne : 0 = identique, plus grand = différent."""
    return float(np.sqrt(np.sum((h1 - h2) ** 2)))


# ---------------------------------------------------------------------------
# 3) Programme principal
# ---------------------------------------------------------------------------
def main():
    dossier = "Images"
    mode = "gray"          # "gray" ou "rgb"
    bins = 256

    # Récupération des images
    extensions = (".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".ppm", ".tif", ".tiff")
    chemins = sorted(
        os.path.join(dossier, f)
        for f in os.listdir(dossier)
        if f.lower().endswith(extensions)
    )

    if not chemins:
        print(f"Aucune image trouvée dans le dossier '{dossier}'.")
        return

    # Calcul des histogrammes
    nom_images = [os.path.basename(c) for c in chemins]
    histos = [histogramme(c, mode=mode, bins=bins) for c in chemins]
    print(f"{len(chemins)} images chargées (mode={mode}, bins={bins}).\n")

    # Comparaison deux à deux
    print("=" * 100)
    print("Matrice de similarité (corrélation) — 1 = identique, proche de 1 = similaire")
    print("=" * 100)

    n = len(nom_images)
    # En-tête
    header = "Image".ljust(16)
    for nom in nom_images:
        header += nom.ljust(14)
    print(header)

    for i in range(n):
        ligne = nom_images[i].ljust(16)
        for j in range(n):
            ligne += f"{correlation(histos[i], histos[j]):.3f}".ljust(14)
        print(ligne)

    # Classement des paires les plus similaires (hors diagonale)
    print("\n" + "=" * 100)
    print("Paires d'images les plus similaires (par corrélation décroissante)")
    print("=" * 100)

    resultats = []
    for i, j in combinations(range(n), 2):
        corr = correlation(histos[i], histos[j])
        resultats.append((corr, nom_images[i], nom_images[j]))

    resultats.sort(reverse=True)
    for corr, a, b in resultats:
        print(f"  corrélation = {corr:.4f}   |   {a}  <->  {b}")

    # Affichage des différentes métriques pour la paire la plus similaire
    print("\n" + "=" * 100)
    print("Métriques détaillées (toutes les métriques, pour chaque paire)")
    print("=" * 100)

    metriques = {
        "corrélation": correlation,
        "chi²": chi_square,
        "intersection": intersection,
        "bhattacharyya": bhattacharyya,
        "hellinger": hellinger,
        "euclidienne": euclidean,
    }

    for i, j in combinations(range(n), 2):
        a, b = nom_images[i], nom_images[j]
        print(f"\n{a}  <->  {b}")
        for nom_metrique, fn in metriques.items():
            print(f"    {nom_metrique:<13}: {fn(histos[i], histos[j]):.6f}")


if __name__ == "__main__":
    main()
