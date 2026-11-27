"""rv2 — boîte à outils d'exécution réelle (MODE A) du prompt système v2
« Retouche & contrôle photo/vidéo immobilière » (Brings Immo Nice / Locazurcasa / Locanissa).

Principe : chaque chiffre produit par cet outil est calculé sur les pixels ou les
métadonnées ([mesuré]) ; les valeurs qui ne peuvent être qu'approchées (température en K,
zones molles) sont étiquetées [estimé] ou « indicatif ». Les points que seul un humain
peut juger (invariants, artefacts, test « visiteur ») sont rendus « À VALIDER (humain) »,
jamais « OK ».
"""

__version__ = "2.0.0"
