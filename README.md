# photo-immo — exécution réelle du prompt v2.1 (retouche & contrôle photo/vidéo immobilière)

Boîte à outils **rv2** : elle exécute en **MODE A** le prompt système
[`PROMPT_v2.1.txt`](PROMPT_v2.1.txt) (Brings Immo Nice — VENTE ; Locazurcasa / Locanissa — LCD).

Elle **améliore la photographie, jamais le bien** :
- toutes les retouches automatiques sont **VERT** et globales ;
- les retouches **ORANGE** reçoivent une mention mesurée sur chaque format ;
- les demandes **ROUGE** sont refusées avec un motif et une alternative conforme ;
- chaque chiffre du rapport est **[mesuré]** (calculé sur les pixels ou les métadonnées) ou **[estimé]**, jamais sans étiquette.

> **Les mentions et les seuils sont une politique interne de l'agence, pas une obligation légale.**
> Pour le cadre juridique (C. conso L121-2, AI Act art. 50 §4, RGPD), voir le §5 du prompt.
> Point vérifié le 27/11/2026 : le Digital Omnibus IA est le règlement (UE) 2026/1744, entré en vigueur le 27/07/2026.
> Son délai au 02/12/2026 ne concerne que l'art. 50 §2 (marquage par les **fournisseurs**), pour les systèmes mis sur le marché
> avant le 02/08/2026 ; l'art. 50 §4 (votre obligation de signaler) s'applique depuis le 02/08/2026.
> Ce point a été vérifié sur des sources secondaires (cabinets d'avocats) ; le texte du JOUE lui-même n'a pas été lu.

---

## 1. Installation

```bash
./install.sh
```

Le script installe exiftool, ffmpeg, Tesseract (fra) et les paquets Python de `requirements.txt` (OpenCV, LibRaw/rawpy, pillow-heif).
Il termine par `python3 -m rv2 selfcheck`, qui doit afficher **« MODE A disponible »**.
Si un élément manque, le selfcheck le nomme : il faut alors travailler en **MODE B** (fiche de réglages), comme le prévoit le §1.

## 2. Utilisation

| Besoin | Commande |
|---|---|
| Vérifier que le MODE A est possible | `python3 -m rv2 selfcheck` |
| Diagnostiquer des photos (JSON étiqueté) | `python3 -m rv2 diagnose IMG_1.dng IMG_2.heic` |
| Traiter une série complète + `RAPPORT.md` (§9) | `python3 -m rv2 run fiche.json` |
| Export conforme seul (remplace `export_immo.py`) | `python3 export_immo.py SOURCE DEST.jpg [qualité]` |
| Vidéo (stabilisation, mention pendant un plan) | `python3 -m rv2 video in.mp4 out.mp4 --mention "Photo retouchée – ciel modifié" --de 12 --a 18` |

### Fiche de travail (`fiche.json`)

Exemple complet : [`exemples/fiche_vente.json`](exemples/fiche_vente.json). Les chemins sont relatifs à la fiche.

| Champ (par photo) | Effet |
|---|---|
| `nn`, `piece`, `fichier` | ordre dans la série, pièce, original (JPG/HEIC/PNG/TIFF 16 bits/RAW/DNG) |
| `brackets` | autres expositions de la **même** prise → fusion (window pull, VERT) ; refusée si décalage > 2 % |
| `redresser` (vrai par défaut) | verticales et perspective, avec les garde-fous du §3 ci-dessous |
| `exposition` | `"auto"` (défaut), une valeur en EV, ou `null` |
| `tons` | `{"shadows", "highlights", "contrast"}` entre 0 et 1, courbe globale |
| `balance_blancs` | `{"methode": "patch", "zone": [x,y,w,h]}` sur une mire de gris = exact ; sinon gray-world partiel |
| `suppressions` | `[{"masque": "m.png", "objet": "câble", "touche_structure": false, "generatif": false}]` |
| `flous` | rectangles `[x,y,w,h]` : n° de rue, plaques, boîtes aux lettres, visages |
| `ecrans_tv` | polygones mis au noir |
| `orange` | `staging`, `travaux`, `ciel`, `saisonnier`, `objet_supprime`, `ia` |
| `rouge_demande` | retouches demandées mais interdites, consignées dans « Retouche refusée + motif » |
| `role` | `photo_vide` (après un staging) ou `etat_reel` (à côté d'une simulation de travaux) |
| `piece_vide`, `exterieur`, `saisonnier_confirme`, `ia_hors_meubles_equipements_etat`, `outil_generatif` | déclarations exigées par le §5 |
| `centre_x` (0–1) | point d'intérêt horizontal pour les recadrages sociaux |

Les masques et rectangles se dessinent **sur l'image après rotation EXIF**, c'est-à-dire telle qu'elle s'affiche.
Le redressement leur applique ensuite exactement la même géométrie.

### Sortie

```
sortie/BIN0421/
├── VENTE_BIN0421_01_SEJOUR_v1.jpg     format portail (3:2 ou 4:3, ≥ 2 400 px si la source le permet)
├── ig45/ ig11/ story/                 mêmes noms, formats sociaux
├── originaux/ORIG_IMG_1001.dng        copie intacte, empreinte SHA-256 vérifiée (§6.6)
├── controle/COMPARE_*.jpg             original | retouche côte à côte (audit des invariants)
├── RAPPORT.md                         livrable §9 : tableau de contrôle + auto-audit + reprises
└── rapport.json                       toutes les mesures
```

## 3. Ce que l'outil fait réellement (MODE A)

| Étape du §6 | Exécuté | Mesuré / garanti |
|---|---|---|
| Lecture | RAW/DNG (LibRaw, 16 bits), HEIC (libheif), JPG/PNG, TIFF 16 bits ; conversion ICC → sRGB (Display P3, Adobe RGB…) **avant** tout changement de mode | P3 (230,20,20) → sRGB (250,0,1) vérifié ; HEIC : rotation `irot` appliquée dans le bon sens et **sans double rotation** avec l'EXIF `Orientation=6` (structure iPhone reproduite sur un fichier de conformité HEIF Nokia, vérifiée à l'œil) |
| Distorsion d'objectif | profil **Lensfun** si boîtier + objectif connus ; sinon **estimation fil à plomb** (k radial qui maximise les lignes droites), **k commun à la série** (même appareil + focale, médiane pondérée) | k injecté retrouvé à ±0,015 ; appliquée seulement si gain ≥ 10 % et cadre perdu ≤ 15 % ; focale < 14 mm équiv. → « à reprendre » (§4) |
| Éléments sensibles | **YOLOv10n** (personnes, écrans, ordinateurs, téléphones, documents) en tuiles à 2 échelles + **YuNet** (visages) | visage d'une personne détectée → **flouté automatiquement** ; visage sans personne (œuvre, affiche) → signalé, non flouté ; état d'écran mesuré contre le mur environnant, « éteint » seulement sur preuve forte, sinon « à vérifier » |
| Diagnostic | % brûlés ≥ 250 / bouchés ≤ 5, inclinaison et convergence des verticales (Hough + régression) avec **indice de fiabilité**, bruit σ (Immerkær), zones molles, qualité JPEG source, focale EXIF, éléments sensibles | température en K toujours **[estimé]** |
| OCR (adresses, plaques, textes injonctifs) | Tesseract multi-passes : image entière + zones de texte détectées, agrandies ×3, CLAHE, **redressement mesuré** par zone, ±6°, négatif | rappel mesuré sur banc de 24 scènes (tailles 18–44 px, ±12°, plaques claires sur fond sombre, faible contraste) : **20/24 (83 %)** contre 8/24 (33 %) en v2.1 |
| Verticales | rotation puis perspective (homographie) | appliquées **seulement** si la détection est fiable, la rotation ≤ 3° et le cadre perdu ≤ 15 % ; sinon géométrie d'origine conservée et photo « à reprendre » |
| Exposition / fusion | EV automatique sans hausse des brûlés > 0,5 pt ; compression douce des hautes lumières ; fusion Mertens alignée MTB | décalages des expositions mesurés en px |
| Balance des blancs | mire de gris = exacte (écart < 1 %) ; automatique = 35 % du gray-world, gains bornés à ±15 % | la teinte des matériaux est un invariant |
| Suppressions | reconstruction classique (OpenCV Telea, **non IA**) | surface mesurée → VERT ≤ 1 % sans élément structurel, sinon ORANGE « Photo retouchée – objet supprimé » |
| Floutage | pixellisation + flou, irréversible | seule la zone ciblée change (vérifié pixel à pixel) |
| Mention ORANGE | une seule mention combinée, incrustée **après** la mise au format | hauteur ≥ 3 % (cible 3,4 %) de **chaque ligne**, dans le cadre, contraste ≥ 4,5:1 ; sinon format non livré |
| Export | JPEG q90, 4:4:4, profil ICC sRGB, **aucune** donnée EXIF, GPS, XMP, IPTC ni commentaire | relu par exiftool après écriture ; fichier supprimé si le contrôle échoue |
| Nommage | `[USAGE]_[BIEN]_[NN]_[PIECE]_v1.jpg` | refus si BIEN/PIECE ressemble à une adresse (VENTE) |
| Conformité | staging/travaux/saisonnier refusés en LCD ; staging uniquement en pièce vide, photo vide en NN+1, jamais en couverture ; travaux à côté de l'état réel ; ciel uniquement en extérieur ; ROUGE toujours refusé avec alternative | — |
| Audit §6.4 | **tous les points calculés OK/ÉCHEC** ; au plus 2 boucles. Invariants : corrélation des gradients, rappel des contours, dérive de teinte locale (ΔE a*b*, dérive globale de balance des blancs retirée par tranche de luminance), modifications hors masques déclarés. Artefacts : halo, postérisation, écrêtage ajouté, bruit amplifié. Test « visiteur » : surface + contact avec une ligne structurelle **mesurés** pour chaque suppression | validé : 5 photos réelles retouchées passent ; halo, postérisation, élément effacé, sol recoloré, surexposition sont tous rejetés ; une zone effacée **déclarée** est acceptée |
| Série | colorimétrie, luminance, focale : photos qui détonnent | — |
| Vidéo | vid.stab en 2 passes, `eq`, rotation sans coins noirs, mention pendant les plans indiqués, `-map_metadata -1` | durée conservée ; GPS et date de tournage absents (exiftool) ; mention présente dans sa fenêtre et absente ailleurs |

## 4. Limites restantes (mesurées, signalées dans chaque rapport)

- **Retouche générative** (ciel, staging, suppression IA) : non intégrée — consignée « À EXÉCUTER par l'opérateur », puis mention apposée.
- **OCR** : 83 % de rappel mesuré — un texte minuscule ou très incliné peut encore échapper ; c'est un filet, pas une garantie.
- **Détection** : un objet très petit ou masqué peut être manqué ; l'état d'un écran vu de profil est souvent « à vérifier » (reflets).
- **Audit calculé** : les critères sont des mesures de substitution, validées sur cas réels et défauts injectés ; leurs seuils sont techniques (§7).
- **Licence YOLOv10n** : poids sous AGPL-3.0 (Ultralytics/THU) — usage interne ; à revoir avant toute redistribution ou mise en service web.
- **DNG via Google Drive** : le connecteur refuse les fichiers > 10 Mo ; les RAW doivent être fournis autrement (dossier local, lien de partage).

## 5. Tests

```bash
tests/fetch_samples.sh                 # DNG public (~48 Mo) + profil Display P3 — non versionnés
python3 -m pytest -q tests/            # 52 tests (dont banc OCR ~1 min ; -m "not slow" pour l'exclure)
```

Les tests couvrent :
- l'iPhone simulé (P3, rotation, GPS, XMP, IPTC, commentaire) ;
- HEIC, DNG réel, CMJN ;
- le refus d'écraser l'original et la copie ORIG_ identique ;
- les mesures d'histogramme ;
- le redressement de ±2,5° et la perspective de 4,4° ramenés à moins de 0,3° ;
- le **refus** de rotation > 3°, de rognage excessif et de correction d'un gros plan non fiable ;
- la balance des blancs sur mire, l'exposition, le bracketing décalé ;
- le floutage et la surface des suppressions ;
- toutes les règles de conformité LCD/VENTE et le nommage ;
- la mention sur chaque format, le pipeline de bout en bout et la vidéo.

## 6. `export_immo.py` : différences avec la version d'origine

La version d'origine est conservée dans [`legacy/`](legacy/export_immo_v2.1_original.py). Mesures sur un banc d'essai réel :

| | Origine | Actuelle |
|---|---|---|
| Rotation EXIF, P3 → sRGB, suppression EXIF/GPS/XMP/IPTC/commentaire, ICC sRGB | ✅ | ✅ |
| HEIC (format natif iPhone) | ❌ plantage `cannot identify image file` | ✅ |
| `SOURCE` = `DEST` | ❌ **l'original est écrasé** (contraire au §6) | ✅ refus |
| RAW/DNG, TIFF 16 bits, CMJN avec profil | ❌ | ✅ |
| Contrôle mesuré du fichier écrit | ❌ | ✅ (exiftool ; fichier supprimé si échec) |

Le défaut de l'origine est figé dans un test (`test_ancien_script_ecrasait_l_original_et_plantait_en_heic`).

## 7. Seuils

| Seuil | Valeur | Nature |
|---|---|---|
| Zone reconstruite VERT | ≤ 1 % | interne agence (§5) |
| Hauteur de la mention | ≥ 3 % | interne agence (§5) ; l'outil vise 3,4 % |
| Rotation maximale | 3° | interne agence (§4) |
| Largeur portail | ≥ 2 400 px | interne agence (§8) ; jamais d'agrandissement |
| Cadre perdu au redressement | ≤ 15 % | technique, outil (≈ perte d'une rotation de 3°) |
| Convergence maximale corrigée | 8° | technique, outil |
| Fiabilité des verticales | ≥ 6 lignes, dispersion ≤ 0,6° | technique, outil |
| Contraste de la mention | ≥ 4,5:1 | référence WCAG AA (lisibilité), pas une obligation légale |
| Décalage maximal entre expositions | 2 % de la largeur | technique, outil |
| Balance des blancs automatique | 35 %, gains ±15 % | technique, outil (invariant §3) |
| Distorsion : gain minimal / cadre perdu max | 10 % de lignes droites / 15 % | technique, outil |
| Focale minimale | 14 mm équiv. | technique, outil (§4 vise 16–24 mm) |
| Audit : corrélation structure / rappel contours | ≥ 0,90 / ≥ 0,85 | technique, outil |
| Audit : dérive de teinte locale (p95) | ΔE a*b* ≤ 6 | technique, outil |
| Audit : halo / niveaux / bruit | ≤ 1,9 / ≥ 0,55 / ≤ 1,5 | technique, outil |
| Audit : modification hors masques déclarés | ≤ 0,5 % | technique, outil |
| Suppression « touche la structure » | > 5 % du masque sur une ligne structurelle | technique, outil |
| Recadrage social maximal | 25 %, au-delà bandes unies | technique, outil |

Pour modifier un seuil : constantes en tête de `rv2/correct.py`, `rv2/export.py` et `rv2/conformity.py`.
