"""
Lecture personnalisée d'un fichier PCX (8 bits/pixel, 1 plan, RLE, palette 256 couleurs)
et affichage avec matplotlib. Aucune bibliothèque d'image (PIL) n'est utilisée pour le décodage.
"""
import struct
import numpy as np
import matplotlib.pyplot as plt


def lire_pcx(chemin):
    with open(chemin, "rb") as f:
        data = f.read()

    # ---------- HEADER (128 bytes) ----------
    fabricant = data[0]
    version = data[1]
    compression = data[2]              # 1 = RLE, 0 = non
    bits_par_pixel = data[3]
    xmin, ymin, xmax, ymax = struct.unpack("<4H", data[4:12])
    nb_plans = data[65]
    octets_par_ligne = struct.unpack("<H", data[66:68])[0]

    if fabricant != 10:
        raise ValueError("Ce n'est pas un fichier PCX (fabricant != 10)")
    if bits_par_pixel != 8 or nb_plans != 1:
        raise NotImplementedError("Seul le PCX 256 couleurs (8 bits, 1 plan) est géré")

    largeur = xmax - xmin + 1
    hauteur = ymax - ymin + 1
    print(f"PCX v{version} : {largeur}x{hauteur}, RLE={compression}, "
          f"{bits_par_pixel} bits/pixel, {octets_par_ligne} octets/ligne")

    # ---------- PALETTE : 768 bytes (256 x RGB) à la fin du fichier ----------
    # Elle est précédée d'un byte marqueur (12 dans la spécification officielle ;
    # l'énoncé indique 192, on accepte donc les deux).
    marqueur = data[-769]
    if marqueur not in (12, 192):
        raise ValueError(f"Marqueur de palette inattendu : {marqueur}")
    pal = np.frombuffer(data[-768:], dtype=np.uint8).reshape(256, 3)  # ordre R,G,B

    # ---------- RASTER (décompression RLE) ----------
    raster_fin = len(data) - 769
    total = octets_par_ligne * hauteur
    pixels = bytearray()
    i = 128
    if compression == 1:
        while len(pixels) < total and i < raster_fin:
            b = data[i]
            i += 1
            if b >= 192:                       # 193..255 (et 192) : octet de répétition
                n = b - 192
                pixels.extend(bytes([data[i]]) * n)
                i += 1
            else:                              # valeur simple
                pixels.append(b)
    else:
        pixels.extend(data[128:128 + total])

    img_index = np.frombuffer(bytes(pixels[:total]), dtype=np.uint8)
    img_index = img_index.reshape(hauteur, octets_par_ligne)[:, :largeur]

    # Conversion indices -> RGB via la palette
    img_rgb = pal[img_index]                   # (hauteur, largeur, 3)
    return img_rgb


if __name__ == "__main__":
    image = lire_pcx("Ex2.pcx")
    plt.imshow(image)
    plt.title("Ex2.pcx")
    plt.axis("off")
    plt.show()
