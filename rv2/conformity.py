"""GARDE-FOU CONFORMITÉ (§5, v2.1). Classe chaque retouche demandée VERT / ORANGE / ROUGE,
refuse les ROUGE (et les ORANGE interdits en LCD), compose la mention ORANGE unique."""
from __future__ import annotations

import re
import unicodedata

VERT_SEUIL_ZONE_PCT = 1.0  # seuil interne §5

MENTIONS = {
    "staging": "Aménagement virtuel – non contractuel",
    "travaux": "Simulation de travaux – non contractuelle",
    "ciel": "Photo retouchée – ciel modifié",
    "saisonnier": "Photo retouchée – état saisonnier",
    "objet_supprime": "Photo retouchée – objet supprimé",
    "ia": "Image modifiée par IA",
}
INTERDIT_LCD = {"staging", "travaux", "saisonnier"}

ROUGE = {
    "effacer_defaut": "effacer un défaut (fissure, humidité, tache, usure…) est interdit",
    "environnement": "effacer/ajouter un élément permanent de l'environnement est interdit",
    "vue": "créer, agrandir, dégager ou remplacer une vue est interdit",
    "ajout_equipement": "ajouter un équipement absent est interdit",
    "volumes": "modifier volumes, hauteur sous plafond ou surface apparente est interdit",
    "etat": "changer l'état réel du bien est interdit",
    "flou_arriere_plan": "ajouter un flou d'arrière-plan est interdit",
    "defloutage_ia": "rendre net par IA une zone floue (détails inventés) est interdit",
    "agrandissement_ia": "agrandir le cadre par IA est interdit",
}
ALTERNATIVES = {
    "effacer_defaut": "montrer le défaut ; le cas échéant photo de détail après réparation réelle",
    "environnement": "cadrer sur la lumière ou l'intérieur plutôt que masquer l'élément",
    "vue": "window pull depuis le bracketing de la même prise de vue, sinon reprise sur place",
    "ajout_equipement": "simulation de travaux ORANGE, en photo supplémentaire (VENTE uniquement), si projet réel",
    "volumes": "focale 16–24 mm équiv., prise depuis un coin",
    "etat": "photo à la bonne saison, ou mention « état saisonnier » si l'état est réellement saisonnier",
    "flou_arriere_plan": "profondeur de champ f/8–f/11, tout net",
    "defloutage_ia": "reprise sur place (trépied, vitesse adaptée)",
    "agrandissement_ia": "reprise sur place à une focale plus courte",
}


def combine_mentions(keys: list[str]) -> str | None:
    """Une seule mention qui nomme toutes les retouches ORANGE (ordre stable, sans doublon)."""
    texts = []
    for k in keys:
        t = MENTIONS[k]
        if t not in texts:
            texts.append(t)
    if not texts:
        return None
    if len(texts) == 1:
        return texts[0]
    # « Photo retouchée – ciel modifié » + « Photo retouchée – objet supprimé » → une seule tête
    heads = [t.split(" – ")[0] for t in texts]
    if all(h == "Photo retouchée" for h in heads):
        return "Photo retouchée – " + ", ".join(t.split(" – ")[1] for t in texts)
    return " · ".join(texts)


def classify_photo(photo: dict, usage: str, all_photos: list[dict]) -> dict:
    """photo : dict de la fiche job. Retourne {'classe', 'orange', 'refus', 'mention'}."""
    refus, orange, notes = [], [], []
    nn = photo["nn"]
    for r in photo.get("rouge_demande", []):
        refus.append({"demande": r, "motif": ROUGE.get(r, "retouche classée ROUGE"),
                      "alternative": ALTERNATIVES.get(r, "—")})

    for k in photo.get("orange", []):
        if k not in MENTIONS:
            refus.append({"demande": k, "motif": "retouche ORANGE inconnue : non exécutée", "alternative": "—"})
            continue
        if usage == "LCD" and k in INTERDIT_LCD:
            refus.append({"demande": k, "motif": f"{k} : 🔴 INTERDIT en LCD (§5)", "alternative": "photo de l'état réel"})
            continue
        if k in ("staging", "travaux") and nn == 1:
            refus.append({"demande": k, "motif": "une photo ORANGE (staging/travaux) n'est jamais la couverture (§8)",
                          "alternative": "placer la photo plus loin dans la série"})
            continue
        if k == "staging":
            vide = next((p for p in all_photos if p["nn"] == nn + 1 and p.get("role") == "photo_vide"), None)
            if not photo.get("piece_vide", False):
                refus.append({"demande": k, "motif": "staging : pièce vide uniquement (§5)", "alternative": "—"})
                continue
            if vide is None:
                refus.append({"demande": k, "motif": "staging : la photo vide doit être livrée juste après (NN+1, role=photo_vide)",
                              "alternative": "ajouter la photo vide en NN+1"})
                continue
        if k == "travaux":
            reel = next((p for p in all_photos if abs(p["nn"] - nn) == 1 and p.get("role") == "etat_reel"), None)
            if reel is None:
                refus.append({"demande": k, "motif": "simulation de travaux : photo supplémentaire, à côté de l'état réel (NN±1, role=etat_reel)",
                              "alternative": "ajouter la photo de l'état réel adjacente"})
                continue
        if k == "ciel" and not photo.get("exterieur", False):
            refus.append({"demande": k, "motif": "ciel : extérieurs uniquement, jamais à travers une fenêtre", "alternative": "—"})
            continue
        if k == "saisonnier" and not photo.get("saisonnier_confirme", False):
            refus.append({"demande": k, "motif": "état saisonnier non confirmé (§2) : sinon c'est changer l'état réel (ROUGE)",
                          "alternative": "confirmer par écrit que l'état est saisonnier et réversible"})
            continue
        if k == "ia" and usage == "LCD" and not photo.get("ia_hors_meubles_equipements_etat", False):
            refus.append({"demande": k, "motif": "LCD : IA admise seulement hors meubles, équipements et état — non confirmé",
                          "alternative": "—"})
            continue
        orange.append(k)

    if photo.get("outil_generatif") and "ia" not in orange and orange:
        notes.append("outil génératif déclaré : la mention nomme la retouche concernée")

    classe = "ROUGE (refus)" if refus and not orange and photo.get("rouge_demande") else ("ORANGE" if orange else "VERT")
    return {"classe": classe, "orange": orange, "refus": refus, "mention": combine_mentions(orange), "notes": notes}


def classify_removal(zone_pct: float, touche_structure: bool, generatif: bool) -> tuple[str, str | None]:
    """§5 : suppression d'objet temporaire. Retourne (classe, clé de mention ORANGE éventuelle)."""
    if zone_pct <= VERT_SEUIL_ZONE_PCT and not touche_structure:
        return "VERT", None
    return "ORANGE", ("ia" if generatif else "objet_supprime")


# ----------------------------------------------------------------- nommage (§8)
_ADDR = re.compile(r"(^|-)(RUE|AVENUE|AV|BD|BOULEVARD|CHEMIN|CH|ALLEE|PLACE|PL|QUAI|IMPASSE|ROUTE|RTE|PROMENADE|"
                   r"PROM|COURS|SQUARE|TRAVERSE|MONTEE|RESIDENCE|\d{5})(-|$)")


def _slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Z0-9-]+", "-", s.upper()).strip("-")


def build_name(usage: str, bien: str, nn: int, piece: str, version: int = 1) -> str:
    usage = usage.upper()
    if usage not in ("VENTE", "LCD"):
        raise ValueError("USAGE doit être VENTE ou LCD")
    b, p = _slug(bien), _slug(piece)
    for label, part in (("BIEN", b), ("PIECE", p)):
        if not part:
            raise ValueError(f"{label} vide")
        if usage == "VENTE" and _ADDR.search(part):
            raise ValueError(f"{label} « {part} » ressemble à une adresse : interdit en VENTE (§5/§8)")
    if not re.fullmatch(r"[A-Z0-9-]{2,24}", b):
        raise ValueError("BIEN : code interne attendu (2–24 caractères A-Z, 0-9, -)")
    return f"{usage}_{b}_{int(nn):02d}_{p}_v{int(version)}.jpg"
