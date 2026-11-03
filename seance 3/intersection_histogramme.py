# -*- coding: utf-8 -*-
"""
Similarité d'images par la méthode d'INTERSECTION d'histogrammes.

L'intersection de deux histogrammes mesure la zone commune entre eux :

    intersection(H1, H2) = Σ min(H1(i), H2(i))

Après normalisation des histogrammes (somme = 1), la valeur obtenue est
comprise entre 0 et 1 :
  - 1.0  -> les histogrammes sont identiques (images très similaires)
  - 0.0  -> aucune intersection (images complètement différentes)

Plus la valeur est proche de 1, plus les images se ressemblent.
"""

import os
from itertools import combinations

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# 1) Calcul de l'histogramme normalisé
# ---------------------------------------------------------------------------
def masque_objet(arr, seuil_fond=40):
    """
    Crée un masque booléen des pixels "objet" (non-fond).

    Les pixels dont l'intensité est en dessous de `seuil_fond` (sur les 3
    canaux) sont considérés comme du fond (noir) et exclus. Cela évite que
    le fond sombre commun à toutes les images ne fausse la comparaison.

    Paramètres
    ----------
    arr : numpy.ndarray (H, W, 3)
        Image RGB.
    seuil_fond : int
        Seuil d'intensité en dessous duquel un pixel est considéré comme fond.

    Retourne
    --------
    numpy.ndarray booléen (H, W) : True pour les pixels objet.
    """
    # Un pixel est "fond" si les 3 canaux sont sous le seuil (pixel sombre).
    est_sombre = np.all(arr < seuil_fond, axis=2)
    return ~est_sombre


def histogramme_normalise(image_path, mode="rgb", bins=256, ignorer_fond=True, seuil_fond=40):
    """
    Calcule l'histogramme normalisé d'une image (somme = 1).

    Paramètres
    ----------
    image_path : str
        Chemin vers l'image.
    mode : str
        "gray" -> niveaux de gris (1 canal).
        "rgb"  -> couleur (3 canaux concaténés).
    bins : int
        Nombre de bins par canal.
    ignorer_fond : bool
        Si True, les pixels sombres (fond) sont exclus de l'histogramme.
    seuil_fond : int
        Seuil d'intensité pour détecter le fond.

    Retourne
    --------
    numpy.ndarray normalisé (somme = 1).
    """
    img = Image.open(image_path)

    if mode == "gray":
        img = img.convert("L")
        arr = np.asarray(img)
        if ignorer_fond:
            masque = arr >= seuil_fond
            arr = arr[masque]
        hist = np.histogram(arr, bins=bins, range=(0, 256))[0]
    elif mode == "rgb":
        img = img.convert("RGB")
        arr = np.asarray(img)
        if ignorer_fond:
            masque = masque_objet(arr, seuil_fond)
            arr = arr[masque]
        canaux = []
        for c in range(3):
            canaux.append(np.histogram(arr[:, c], bins=bins, range=(0, 256))[0])
        hist = np.concatenate(canaux)
    else:
        raise ValueError("mode doit être 'gray' ou 'rgb'")

    total = hist.sum()
    if total > 0:
        hist = hist.astype(float) / total
    return hist


def histogramme_par_blocs(image_path, mode="rgb", bins=64, grille=(4, 4),
                          ignorer_fond=True, seuil_fond=40):
    """
    Calcule un descripteur d'histogramme par blocs (spatial pyramid).

    L'image est découpée en une grille de `grille` blocs. Pour chaque bloc,
    on calcule un histogramme de couleur (en excluant le fond), puis on
    concatène tous les histogrammes en un seul vecteur normalisé globalement.

    Paramètres
    ----------
    image_path : str
        Chemin vers l'image.
    mode : str
        "gray" ou "rgb".
    bins : int
        Nombre de bins par canal et par bloc.
    grille : tuple (lignes, colonnes)
        Découpage de l'image en blocs.
    ignorer_fond : bool
        Exclure les pixels sombres (fond).
    seuil_fond : int
        Seuil d'intensité pour détecter le fond.

    Retourne
    --------
    numpy.ndarray (concaténation des histogrammes de blocs, somme = 1).
    """
    img = Image.open(image_path)
    if mode == "rgb":
        img = img.convert("RGB")
    else:
        img = img.convert("L")

    arr = np.asarray(img)
    h, w = arr.shape[:2]
    lignes, colonnes = grille

    hauteur_bloc = h // lignes
    largeur_bloc = w // colonnes

    descripteurs = []
    for i in range(lignes):
        for j in range(colonnes):
            bloc = arr[i * hauteur_bloc:(i + 1) * hauteur_bloc,
                        j * largeur_bloc:(j + 1) * largeur_bloc]

            if ignorer_fond:
                if mode == "rgb":
                    masque = masque_objet(bloc, seuil_fond)
                    bloc = bloc[masque]
                else:
                    bloc = bloc[bloc >= seuil_fond]

            if bloc.size == 0:
                # Bloc entièrement du fond : histogramme vide (zéros).
                if mode == "rgb":
                    hist_bloc = np.zeros(bins * 3)
                else:
                    hist_bloc = np.zeros(bins)
            else:
                if mode == "rgb":
                    hist_bloc = []
                    for c in range(3):
                        hist_bloc.append(
                            np.histogram(bloc[:, c], bins=bins, range=(0, 256))[0]
                        )
                    hist_bloc = np.concatenate(hist_bloc)
                else:
                    hist_bloc = np.histogram(bloc, bins=bins, range=(0, 256))[0]

            descripteurs.append(hist_bloc.astype(float))

    descripteur = np.concatenate(descripteurs)
    total = descripteur.sum()
    if total > 0:
        descripteur = descripteur / total
    return descripteur


# ---------------------------------------------------------------------------
# 2) Intersection d'histogrammes
# ---------------------------------------------------------------------------
def intersection(h1, h2):
    """
    Intersection de deux histogrammes normalisés.

    Retourne une valeur entre 0 et 1 :
      1.0 = identiques, 0.0 = aucune intersection.
    """
    return float(np.sum(np.minimum(h1, h2)))


# ---------------------------------------------------------------------------
# 3) Programme principal
# ---------------------------------------------------------------------------
def main():
    dossier = "Images"
    mode = "rgb"       # "rgb" (couleur) ou "gray" (niveaux de gris)

    # --- Paramètres de la méthode ---
    # "global" : histogramme global (ignore la position des couleurs)
    # "blocs"  : histogramme par blocs (préserve la position spatiale, plus fiable)
    methode = "blocs"

    ignorer_fond = True   # exclut les pixels sombres (fond) de l'histogramme
    seuil_fond = 40       # intensité < 40 sur les 3 canaux => considéré comme fond

    bins = 256          # bins pour la méthode globale
    bins_blocs = 64     # bins par canal pour la méthode par blocs
    grille = (4, 4)     # découpage en 4x4 = 16 blocs

    extensions = (".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".ppm", ".tif", ".tiff")
    chemins = sorted(
        os.path.join(dossier, f)
        for f in os.listdir(dossier)
        if f.lower().endswith(extensions)
    )

    if not chemins:
        print(f"Aucune image trouvée dans le dossier '{dossier}'.")
        return

    nom_images = [os.path.basename(c) for c in chemins]

    if methode == "blocs":
        histos = [
            histogramme_par_blocs(c, mode=mode, bins=bins_blocs, grille=grille,
                                  ignorer_fond=ignorer_fond, seuil_fond=seuil_fond)
            for c in chemins
        ]
        print(f"{len(chemins)} images chargées "
              f"(méthode=blocs, mode={mode}, grille={grille}, bins/bloc={bins_blocs}, "
              f"ignorer_fond={ignorer_fond}, seuil={seuil_fond}).\n")
    else:
        histos = [
            histogramme_normalise(c, mode=mode, bins=bins,
                                  ignorer_fond=ignorer_fond, seuil_fond=seuil_fond)
            for c in chemins
        ]
        print(f"{len(chemins)} images chargées "
              f"(méthode=globale, mode={mode}, bins={bins}, "
              f"ignorer_fond={ignorer_fond}, seuil={seuil_fond}).\n")

    n = len(nom_images)

    # --- Matrice d'intersection ---
    print("=" * 90)
    print("Matrice d'intersection — 1 = identique, proche de 1 = similaire")
    print("=" * 90)

    header = "Image".ljust(16)
    for nom in nom_images:
        header += nom.ljust(14)
    print(header)

    for i in range(n):
        ligne = nom_images[i].ljust(16)
        for j in range(n):
            ligne += f"{intersection(histos[i], histos[j]):.3f}".ljust(14)
        print(ligne)

    # --- Classement des paires les plus similaires ---
    print("\n" + "=" * 90)
    print("Paires d'images les plus similaires (par intersection décroissante)")
    print("=" * 90)

    resultats = []
    for i, j in combinations(range(n), 2):
        inter = intersection(histos[i], histos[j])
        resultats.append((inter, nom_images[i], nom_images[j]))

    resultats.sort(reverse=True)

    for rang, (inter, a, b) in enumerate(resultats, start=1):
        # Pourcentage de similarité
        pct = inter * 100.0
        print(f"  {rang:>2}. intersection = {inter:.4f} ({pct:5.1f}%)   |   {a}  <->  {b}")

    # --- Seuil de similarité ---
    seuil = 0.70
    print("\n" + "=" * 90)
    print(f"Paires considérées comme SIMILAIRES (intersection >= {seuil})")
    print("=" * 90)

    similaires = [(inter, a, b) for inter, a, b in resultats if inter >= seuil]
    if similaires:
        for inter, a, b in similaires:
            print(f"  intersection = {inter:.4f} ({inter * 100:.1f}%)   |   {a}  <->  {b}")
    else:
        print("  Aucune paire ne dépasse ce seuil.")


if __name__ == "__main__":
    main()
