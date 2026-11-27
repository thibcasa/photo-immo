"""LIVRABLE OBLIGATOIRE (§9) au format Markdown."""
from __future__ import annotations


def _v(d: dict, key: str) -> str:
    x = d.get(key)
    if not isinstance(x, dict):
        return "—"
    val = x.get("valeur")
    if val is None:
        return f"n/d [{x.get('source', '?')}]"
    src = "mesuré" if x.get("source", "").startswith("mesuré") else "estimé"
    return f"{val} [{src}]"


def _cell(s: str) -> str:
    return str(s).replace("|", "/").replace("\n", " ")


def render_markdown(res: dict) -> str:
    L = [f"**Mode {res['mode']} · Usage {res['usage']} · Bien {res['bien']} · Hypothèses : "
         + ("; ".join(res["hypotheses"]) if res["hypotheses"] else "aucune") + "**", ""]
    L.append("| N° photo | Diagnostic (chiffres étiquetés) | Retouches effectuées / prescrites | Classe | Outil (si IA) "
             "| Mention apposée | Retouche refusée + motif | À reprendre sur place ? | Statut ✅/⚠️ |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in res["photos"]:
        if "erreur" in r:
            L.append(f"| {r['nn']:02d} {r['piece']} | — | — | — | — | — | — | — | ⚠️ erreur : {_cell(r['erreur'])} |")
            continue
        d = r["diagnostic"]
        diag = (f"brûlés {_v(d, 'brules_pct')} % · bouchés {_v(d, 'bouches_pct')} % · "
                f"≈{_v(d, 'temperature_K')} K · inclinaison {_v(d, 'inclinaison_deg')}° · "
                f"convergence {_v(d, 'convergence_bords_deg')}° (fiable : {_v(d, 'verticales_fiables')}) · bruit σ {_v(d, 'bruit_sigma')} · "
                f"zones molles {_v(d, 'zones_molles')}")
        tv = d.get("texte_visible", {})
        if tv.get("adresse_suspectee") or tv.get("plaque_suspectee"):
            diag += " · ⚠ texte d'adresse/plaque détecté (OCR)"
        refus = "; ".join(f"{x['demande']} : {x['motif']} → {x['alternative']}" for x in r["refus"]) or "—"
        rep = "; ".join(r["a_reprendre"]) or "Non"
        statut = r["statut"]
        if r.get("points_humains"):
            statut += f" (contrôles auto) · à cocher : {', '.join(r['points_humains'])}"
        if r.get("alertes"):
            statut += " · " + " ; ".join(r["alertes"])
        L.append(f"| {r['nn']:02d} {r['piece']} — `{r['fichier']}` | {_cell(diag)} | {_cell('; '.join(r['retouches']))} "
                 f"| {r['classe']} | {_cell(r['outil_ia'])} | {_cell(r['mention'])} | {_cell(refus)} | {_cell(rep)} | {_cell(statut)} |")
    L += ["", "### Auto-audit §6.4 (par photo)"]
    for r in res["photos"]:
        if r.get("audit"):
            L.append(f"- **{r['nn']:02d}** : " + " · ".join(f"{k} = {v}" for k, v in r["audit"].items())
                     + f" · boucles : {r.get('boucles', 1)} · comparaison : `{r.get('comparaison', '—')}`")
            for e in r.get("exports", []):
                m = e.get("mention")
                extra = f", mention {m['hauteur_texte_pct']} % / contraste {m['contraste']}" if m else ""
                L.append(f"  - {e['format']} : {'livré' if e['livrable'] else 'NON livré — ' + e.get('motif', '')}"
                         f" ({e['mise_au_format'].get('mode', '?')}{extra})")
    if res.get("coherence_serie"):
        L += ["", "### Cohérence de la série"] + [f"- {x}" for x in res["coherence_serie"]]
    reprises = [r for r in res["photos"] if r.get("a_reprendre")]
    L += ["", "### À refaire sur place"]
    if not reprises:
        L.append("Aucune.")
    for r in reprises[:3]:
        L.append(f"- {r['nn']:02d} {r['piece']} : {'; '.join(r['a_reprendre'])} → appareil de niveau à 1,20–1,50 m, "
                 "16–24 mm équiv., depuis un coin, f/8–f/11 sur trépied, bracketing ±2 EV, lumières allumées, pièce rangée.")
    return "\n".join(L) + "\n"
