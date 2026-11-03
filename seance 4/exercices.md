# TP – Recherche et Similarité d'Images

## Exercice 1 : Similarité basée sur les couleurs dominantes

### Objectif
Représenter chaque image par ses couleurs dominantes, puis calculer la similarité entre les images à partir de leurs caractéristiques colorimétriques.

### Travail demandé
1. Reprendre le matériel de l'atelier sur le calcul de similarité (voir le code fourni en pièce jointe).
2. Extraire les couleurs dominantes de chaque image.
3. Ne retenir que les couleurs dominantes couvrant ensemble **90 % des pixels** de l'image.
4. Proposer une mesure de distance entre deux images qui tient compte du fait que le nombre de couleurs dominantes peut être différent d'une image à l'autre.
5. Implémenter la méthode sur le jeu d'images utilisé dans l'atelier.
6. Calculer la matrice de similarité entre les images.
7. Analyser et discuter les résultats obtenus.

---

## Exercice 2 : Similarité basée sur les histogrammes de couleur

### Objectif
Représenter chaque image à l'aide de ses histogrammes de couleur normalisés et mesurer la similarité entre les images.

### Travail demandé
1. Reprendre le matériel de l'atelier sur le calcul de similarité (voir le code fourni en pièce jointe).
2. Calculer les histogrammes de couleur de chaque image.
3. Normaliser les histogrammes obtenus.
4. Utiliser **l'intersection d'histogrammes** comme mesure de similarité entre deux images.
5. Implémenter la méthode sur le jeu d'images utilisé dans l'atelier.
6. Calculer la matrice de similarité entre les images.
7. Analyser et discuter les résultats obtenus.

---

## Exercice 3 : Recherche d'images par les caractéristiques de Tamura

### Objectif
Mettre en place un système de recherche d'images basé sur les caractéristiques de texture de Tamura.

### Travail demandé

1. Charger le dossier partagé contenant les images de texture.
2. Pour chaque image, extraire les quatre caractéristiques de Tamura :
   - **Rudesse (Coarseness)** ;
   - **Contraste (Contrast)** ;
   - **Directionnalité (Directionality)** ;
   - **Granularité (Granularity)**.
3. Créer un fichier JSON pour chaque image, portant le même nom (même radical) que l'image originale.
4. Enregistrer dans chaque fichier JSON le vecteur des caractéristiques de Tamura correspondant à l'image.

### Recherche par similarité

1. Choisir une image du dossier comme image requête.
2. Calculer son vecteur de caractéristiques de Tamura.
3. Comparer ce vecteur avec ceux des autres images du dossier.
4. Calculer la distance entre l'image requête et chaque image.
5. Retourner les **5 images les plus similaires**, triées par ordre de similarité décroissante.
6. Afficher les cinq images retrouvées avec leur distance respective en légende.

---

## Résultats attendus

Pour chaque exercice, fournir :

- Le code source Python.
- Les résultats de l'exécution.
- Les matrices de similarité pour les exercices 1 et 2.
- Les fichiers JSON contenant les caractéristiques de Tamura pour l'exercice 3.
- Les visualisations des résultats de recherche.
- Une courte discussion et une interprétation des résultats obtenus.

## Technologies et bibliothèques

- Python
- OpenCV
- NumPy
- Matplotlib
- JSON
- Scikit-learn (si nécessaire)

---

**Fin du TP**