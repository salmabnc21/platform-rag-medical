from langchain_community.llms import Ollama
from duckduckgo_search import DDGS
import re

# ?? LLM ??????????????????????????????????????????????????????????????????????
llm = Ollama(
    model="cniongolo/biomistral",
    base_url="http://localhost:11434",
    temperature=0.0,
    top_p=0.9,
    top_k=40,
)

# ?? Abreviations medicales ????????????????????????????????????????????????????
_ABBREV = [
    (r'\bHb\b',    'Hemoglobine (Hb)'),
    (r'\bHGB\b',   'Hemoglobine (HGB)'),
    (r'\bGR\b',    'Hematies / Globules rouges (GR)'),
    (r'\bRBC\b',   'Hematies / erythrocytes (RBC)'),
    (r'\bGB\b',    'Leucocytes / Globules blancs (GB)'),
    (r'\bWBC\b',   'Leucocytes (WBC)'),
    (r'\bPLT\b',   'Plaquettes (PLT)'),
    (r'\bHCT\b',   'Hematocrite (HCT)'),
    (r'\bVGM\b',   'Volume globulaire moyen (VGM)'),
    (r'\bMCV\b',   'Volume corpusculaire moyen (MCV)'),
    (r'\bTCMH\b',  'Teneur corpusculaire en hemoglobine (TCMH)'),
    (r'\bMCH\b',   'Teneur corpusculaire en hemoglobine (MCH)'),
    (r'\bCCMH\b',  'Concentration corpusculaire en hemoglobine (CCMH)'),
    (r'\bMCHC\b',  'Concentration corpusculaire en hemoglobine (MCHC)'),
    (r'\bIDR\b',   'Indice de distribution des reticulocytes (IDR)'),
    (r'\bRDW\b',   'Distribution des globules rouges (RDW)'),
    (r'\bVS\b',    'Vitesse de sedimentation (VS)'),
    (r'\bCRP\b',   'Proteine C-reactive (CRP)'),
    (r'\bNFS\b',   'Numeration formule sanguine (NFS)'),
    (r'\bTP\b',    'Taux de prothrombine (TP)'),
    (r'\bINR\b',   'Rapport normalise international (INR)'),
    (r'\bTCA\b',   'Temps de cephaline activee (TCA)'),
    (r'\bGly\b',   'Glycemie (Gly)'),
    (r'\bHbA1c\b', 'Hemoglobine glyquee (HbA1c)'),
]


def _normalize_medical_context(text: str) -> str:
    for pattern, replacement in _ABBREV:
        text = re.sub(pattern, replacement, text)
    return text


# ?? Extraction deterministe des valeurs de bilan ??????????????????????????????
_RE_NORM_RANGE = re.compile(r'N[z:]?\s*([\d,. ]+?)\s*[-?]\s*([\d,.]+)', re.IGNORECASE)
_RE_NORM_LT    = re.compile(r'N[z:]?\s*<\s*([\d,.]+)', re.IGNORECASE)
_RE_NORM_GT    = re.compile(r'N[z:]?\s*>\s*([\d,.]+)', re.IGNORECASE)
_RE_NUM        = re.compile(r'\d+(?:[. ]\s*\d+)*(?:[,.]\d+)?')
_RE_NORM_START = re.compile(r'\s+(N[z:]?\s*[<>]?\s*\d.*)', re.IGNORECASE)


def _bilan_parse_float(s: str) -> float:
    cleaned = re.sub(r'\s', '', s).replace('\xa0', '')
    parts = cleaned.split('.')
    if len(parts) > 2:
        cleaned = ''.join(parts)
    return float(cleaned.replace(',', '.'))


def _bilan_extract_norm(seg: str):
    m = _RE_NORM_RANGE.search(seg)
    if m:
        return (_bilan_parse_float(m.group(1)), _bilan_parse_float(m.group(2)), 'range')
    m = _RE_NORM_LT.search(seg)
    if m:
        return (None, _bilan_parse_float(m.group(1)), 'lt')
    m = _RE_NORM_GT.search(seg)
    if m:
        return (_bilan_parse_float(m.group(1)), None, 'gt')
    return None


def _bilan_norm_str(norm) -> str:
    if norm is None:
        return "sans norme"
    low, high, nt = norm
    if nt == 'range':
        return f"{low} - {high}"
    if nt == 'lt':
        return f"< {high}"
    if nt == 'gt':
        return f"> {low}"
    return "?"


# ?? Statuts biologiques ? 5 niveaux (calcul purement Python, sans LLM) ???????
# Seuils critiques relatifs aux bornes normales
_CRITICAL_LOW_RATIO  = 0.70   # valeur < 70 % de la borne inferieure -> CRITICAL_LOW
_CRITICAL_HIGH_RATIO = 1.50   # valeur > 150 % de la borne superieure -> CRITICAL_HIGH

# Libelles affiches a l'utilisateur
_STATUS_LABEL = {
    "NORMAL":        "NORMAL",
    "LOW":           "BAS",
    "HIGH":          "eLEVe",
    "CRITICAL_LOW":  "CRITIQUE-BAS",
    "CRITICAL_HIGH": "CRITIQUE-eLEVe",
    "NON_EVALUE":    "NON eVALUe",
}


def _bilan_status(value: float, norm) -> str:
    """Calcule le statut biologique sur 5 niveaux. Entierement deterministe, sans LLM."""
    if norm is None:
        return "NON_EVALUE"
    low, high, nt = norm

    # Correction d'unite si le ratio valeur/borne est aberrant (ex : /mm? vs /?L)
    cmp = value
    ref = high if high is not None else low
    if ref and ref > 0:
        ratio = value / ref
        if ratio > 500_000:
            cmp = value / 1_000_000
        elif ratio > 500:
            cmp = value / 1_000

    if nt == 'range':
        if low is not None and cmp < low * _CRITICAL_LOW_RATIO:
            return "CRITICAL_LOW"
        if low is not None and cmp < low:
            return "LOW"
        if high is not None and cmp > high * _CRITICAL_HIGH_RATIO:
            return "CRITICAL_HIGH"
        if high is not None and cmp > high:
            return "HIGH"
        return "NORMAL"

    if nt == 'lt':
        if cmp >= high * _CRITICAL_HIGH_RATIO:
            return "CRITICAL_HIGH"
        if cmp >= high:
            return "HIGH"
        return "NORMAL"

    if nt == 'gt':
        if cmp <= low * _CRITICAL_LOW_RATIO:
            return "CRITICAL_LOW"
        if cmp <= low:
            return "LOW"
        return "NORMAL"

    return "NON_EVALUE"


_RE_EXP_UNIT = re.compile(r'(\d)\s+(\d+)\^')   # "185 10^9/L" -> neutralise le coefficient


def _parse_bilan_line(line: str):
    """Parse une ligne de bilan et retourne un dict avec label, valeur, unite, norme et statut."""
    line = line.strip()
    if not line:
        return None
    nm = _RE_NORM_START.search(line)
    norm_part = nm.group(1).strip() if nm else ''
    data_part = line[:nm.start()].strip() if nm else line
    # Neutralise la notation exponentielle d'unite (ex: "185 10^9/L" -> "185 X^9/L")
    # pour que le regex numerique n'absorbe pas le coefficient de l'unite.
    data_part_clean = _RE_EXP_UNIT.sub(r'\1 X^', data_part)
    vm = _RE_NUM.search(data_part_clean)
    if not vm:
        return None
    raw_val = vm.group(0)
    try:
        value = _bilan_parse_float(raw_val)
    except Exception:
        return None
    label = data_part[:vm.start()].strip().rstrip(' .:,/')
    if not label:
        return None
    unit = data_part[vm.end():].strip().strip('.,;:')
    norm = _bilan_extract_norm(norm_part) if norm_part else None
    return {
        "label":  label,
        "value":  value,
        "unit":   unit,
        "norm":   _bilan_norm_str(norm),
        "status": _bilan_status(value, norm),
    }


def extract_bilan_values(ocr_text: str) -> list:
    """Extrait les valeurs biologiques de maniere deterministe. Aucun LLM implique."""
    return [r for r in (_parse_bilan_line(l) for l in ocr_text.splitlines()) if r]


# ?? Tableau structure pour le LLM ?????????????????????????????????????????????
def _build_bilan_table(rows: list) -> str:
    """Formate le tableau de bilan avec statuts pre-calcules pour l'injecter dans le prompt."""
    header = (
        f"{'PARAMeTRE':<30} {'VALEUR':>10} {'UNITe':<12} {'ReFeRENCE':<20} STATUT CALCULe"
    )
    separator = "-" * 95
    lines = [header, separator]
    for r in rows:
        param  = r["label"][:29].ljust(30)
        val    = str(r["value"]).rjust(10)
        unit   = (r.get("unit") or "-")[:11].ljust(12)
        ref    = r["norm"][:19].ljust(20)
        status = _STATUS_LABEL.get(r["status"], r["status"])
        lines.append(f"{param} {val} {unit} {ref} {status}")
    return "\n".join(lines)


# ?? Prompt medical structure ??????????????????????????????????????????????????
_MEDICAL_PROMPT = """\
Tu es un biologiste medical expert. Analyse le bilan biologique fourni et produis \
une synthese medicale structuree en fran?ais.

IMPORTANT : Les statuts biologiques ci-dessous ont ete calcules de fa?on \
DeTERMINISTE par un algorithme Python fiable. Tu dois les respecter \
SCRUPULEUSEMENT. Tu ne peux pas les modifier ni les contredire.

=== TABLEAU DU BILAN (statuts pre-calcules, non modifiables) ===
{bilan_table}
===============================================================

ReGLES ABSOLUES ? violation = reponse incorrecte :
1. Un parametre NORMAL ne peut JAMAIS etre decrit comme bas, eleve ou anormal.
2. Un parametre NORMAL n'autorise AUCUNE hypothese de maladie liee a ce parametre.
   -> Hemoglobine NORMALE : interdiction totale de mentionner anemie ou transfusion.
   -> Leucocytes NORMAUX : interdiction de mentionner leucopenie ou infection severe.
   -> Plaquettes NORMALES : interdiction de mentionner thrombopenie ou risque hemorragique.
3. Les hypotheses cliniques ne concernent QUE les parametres BAS, eLEVe, \
CRITIQUE-BAS ou CRITIQUE-eLEVe.
4. Un traitement invasif (transfusion, chimiotherapie, etc.) n'est propose que si \
un statut CRITIQUE-BAS ou CRITIQUE-eLEVe le justifie explicitement.
5. Ne jamais inventer une valeur, une maladie ou un traitement absent du tableau.

Produis ta reponse en respectant EXACTEMENT cette structure en 6 sections :

## Resume du bilan
Synthese globale en 2 a 3 phrases simples, comprehensibles par un patient.
Exemple : "Le bilan est globalement rassurant. La majorite des parametres sont \
dans les valeurs normales."

## Parametres normaux
Liste chaque parametre dont le statut calcule est NORMAL, avec sa valeur entre \
parentheses. Un parametre par ligne.

## Parametres anormaux
Liste chaque parametre dont le statut est BAS, eLEVe, CRITIQUE-BAS ou \
CRITIQUE-eLEVe, avec une explication breve de la signification clinique de \
l'anomalie.
Si aucun parametre anormal : ecrire exactement "Aucune anomalie detectee."

## Signes de gravite
Mentionne uniquement les parametres dont le statut est CRITIQUE-BAS ou \
CRITIQUE-eLEVe, en expliquant pourquoi c'est grave.
Si aucun parametre critique : ecrire exactement \
"Aucun signe biologique de gravite immediate n'est identifie sur ce bilan."

## Interpretation clinique
Propose des hypotheses medicales compatibles UNIQUEMENT avec les anomalies \
reellement presentes dans le tableau.
Si tout est NORMAL : ecrire exactement \
"Aucune anomalie biologique significative ne justifie une hypothese clinique \
particuliere."

## Conduite a tenir
Propose des examens complementaires, une consultation ou une surveillance \
UNIQUEMENT si les donnees biologiques le justifient.
Si tout est NORMAL : ecrire exactement \
"Aucune action urgente requise. Un suivi de routine est recommande."
"""


def build_llm_prompt(rows: list) -> str:
    return _MEDICAL_PROMPT.format(bilan_table=_build_bilan_table(rows))


def _build_synthesis_response(rows: list) -> str:
    """Synthese structuree entierement deterministe — aucun LLM implique."""
    normal    = [r for r in rows if r["status"] == "NORMAL"]
    low       = [r for r in rows if r["status"] == "LOW"]
    high      = [r for r in rows if r["status"] == "HIGH"]
    crit_low  = [r for r in rows if r["status"] == "CRITICAL_LOW"]
    crit_high = [r for r in rows if r["status"] == "CRITICAL_HIGH"]
    anomaux   = low + high + crit_low + crit_high
    critiques = crit_low + crit_high

    lines = ["## Resume du bilan"]
    if not anomaux:
        lines.append(
            "Le bilan est globalement normal. "
            "Tous les parametres analyses sont dans les valeurs de reference."
        )
    else:
        lines.append(
            f"Le bilan presente {len(anomaux)} parametre(s) en dehors des valeurs de reference."
        )

    if normal:
        lines += ["", "## Parametres normaux"]
        for r in normal:
            unit_str = f" {r['unit']}" if r.get("unit") else ""
            lines.append(f"- {r['label']} : {r['value']}{unit_str} (ref: {r['norm']})")

    lines += ["", "## Parametres anormaux"]
    if anomaux:
        for r in anomaux:
            unit_str  = f" {r['unit']}" if r.get("unit") else ""
            status_fr = _STATUS_LABEL.get(r["status"], r["status"])
            lines.append(
                f"- {r['label']} : {r['value']}{unit_str} — {status_fr} (ref: {r['norm']})"
            )
    else:
        lines.append("Aucune anomalie detectee.")

    lines += ["", "## Signes de gravite"]
    if critiques:
        for r in critiques:
            unit_str  = f" {r['unit']}" if r.get("unit") else ""
            status_fr = _STATUS_LABEL.get(r["status"], r["status"])
            lines.append(
                f"- {r['label']} : {r['value']}{unit_str} — {status_fr} (ref: {r['norm']})"
            )
    else:
        lines.append(
            "Aucun signe biologique de gravite immediate n'est identifie sur ce bilan."
        )

    return "\n".join(lines)


# ?? Garde-fous anti-hallucination ?????????????????????????????????????????????
# Chaque regle : si le parametre identifie par param_keys est NORMAL,
# les termes forbidden_terms ne doivent pas apparaitre dans la reponse.
_GUARDRAIL_RULES = [
    {
        "param_keys":      ["hemoglobin", "hemoglobin", "hgb"],
        "forbidden_terms": [
            "anemie", "anemie", "transfusion", "culot globulaire",
            "concentre de globules rouges", "hemoglobine basse",
            "hemoglobine basse", "deficit en fer", "carence martiale",
        ],
    },
    {
        "param_keys":      ["leucocyt", "globules blancs", "wbc"],
        "forbidden_terms": [
            "leucopenie", "leucopenie", "agranulocytose",
            "neutropenie", "neutropenie", "infection bacterienne severe",
        ],
    },
    {
        "param_keys":      ["plaquett", "plt", "thrombocyt"],
        "forbidden_terms": [
            "thrombopenie", "thrombopenie", "purpura",
            "risque hemorragique", "risque hemorragique",
            "hemorragie spontanee", "hemorragie spontanee",
        ],
    },
    {
        "param_keys":      ["glycemie", "glycemie", "glucose"],
        "forbidden_terms": [
            "hypoglycemie", "hypoglycemie",
            "hyperglycemie severe", "hyperglycemie severe",
            "diabete decompense", "diabete decompense",
        ],
    },
    {
        "param_keys":      ["creatinin", "creatinin"],
        "forbidden_terms": [
            "insuffisance renale severe", "insuffisance renale severe",
            "dialyse", "uremie", "uremie",
        ],
    },
]

# Mots signalant une anomalie, utilises pour la verification generique
_ANOMALY_WORDS = [
    "bas", "basse", "insuffisant", "insuffisante",
    "deficit", "deficit", "diminue", "diminue",
    "reduit", "reduit", "abaisse", "abaisse",
    "eleve", "eleve", "augmente", "augmente",
    "excessif", "excessive", "anormal", "anormale",
    "pathologique", "critique",
]


def _find_row_status(rows: list, keys: list):
    for row in rows:
        label_lower = row["label"].lower()
        if any(k in label_lower for k in keys):
            return row["status"]
    return None


def validate_response(response: str, rows: list) -> tuple:
    """
    Verifie la coherence entre la reponse LLM et les statuts calcules.
    Retourne (is_valid: bool, violations: list[str]).
    """
    violations = []
    resp_lower = response.lower()

    # Regles specifiques : termes interdits si le parametre est NORMAL
    for rule in _GUARDRAIL_RULES:
        status = _find_row_status(rows, rule["param_keys"])
        if status == "NORMAL":
            for term in rule["forbidden_terms"]:
                if term.lower() in resp_lower:
                    param_display = rule["param_keys"][0]
                    violations.append(
                        f"Contradiction : '{term}' mentionne alors que "
                        f"le parametre '{param_display}' est NORMAL"
                    )

    # Verification generique : mot d'anomalie a proximite d'un parametre NORMAL
    for row in rows:
        if row["status"] != "NORMAL":
            continue
        label_short = row["label"].lower()[:8]
        for word in _ANOMALY_WORDS:
            pat = re.compile(
                rf'{re.escape(label_short)}.{{0,80}}{re.escape(word)}',
                re.IGNORECASE,
            )
            if pat.search(resp_lower):
                violations.append(
                    f"Contradiction possible : '{row['label']}' est NORMAL "
                    f"mais '{word}' apparait a proximite dans la reponse"
                )

    return len(violations) == 0, violations


def _append_validation_warning(response: str, violations: list) -> str:
    if not violations:
        return response
    warning = (
        "\n\n---\n**AVERTISSEMENT DE COHeRENCE AUTOMATIQUE**\n"
        "Les incoherences suivantes ont ete detectees entre la reponse generee "
        "et les statuts biologiques calcules :\n"
    )
    for v in violations:
        warning += f"- {v}\n"
    warning += (
        "\nFiez-vous EXCLUSIVEMENT aux statuts du tableau de bilan pour "
        "l'interpretation definitive. Consultez un professionnel de sante."
    )
    return response + warning


# ?? Nettoyage OCR ?????????????????????????????????????????????????????????????
_RE_OCR_TILDE      = re.compile(r'~~+')
_RE_OCR_EMDASH_UND = re.compile(r'[??]\s*_+')
_RE_OCR_MULTI_UND  = re.compile(r'_{2,}')
_RE_OCR_LONE_STAR  = re.compile(r'(?<!\w)\*(?!\w)')
_RE_OCR_LONE_PIPE  = re.compile(r'(?<!\w)\|(?!\w)')
_RE_OCR_LONE_EQ    = re.compile(r'(?<!\w)=(?!\w)')
_RE_OCR_EQ_CAPS    = re.compile(r'=(?=[A-Z]{2})')
_RE_MULTI_SPACE    = re.compile(r'[ \t]{2,}')
_RE_SEPARATOR_LINE = re.compile(r'^[\s\-=_|~*#\.]{3,}$')
_RE_HAS_DIGIT      = re.compile(r'\d')


def _clean_ocr_text(text: str) -> str:
    text = _RE_OCR_TILDE.sub('', text)
    text = _RE_OCR_EMDASH_UND.sub('', text)
    text = _RE_OCR_MULTI_UND.sub('', text)
    text = _RE_OCR_LONE_STAR.sub('', text)
    text = _RE_OCR_LONE_PIPE.sub('', text)
    text = _RE_OCR_LONE_EQ.sub('', text)
    text = _RE_OCR_EQ_CAPS.sub('', text)
    text = _RE_MULTI_SPACE.sub(' ', text)
    useful = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if _RE_SEPARATOR_LINE.match(line):
            continue
        if _RE_HAS_DIGIT.search(line):
            useful.append(line)
    return '\n'.join(useful)


# ?? Detection requete de synthese ?????????????????????????????????????????????
_SYNTHESIS_KEYWORDS = [
    "resume", "resume", "synthese", "synthese", "ensemble",
    "toutes les valeurs", "vue d ensemble", "vue d'ensemble",
    "toutes les analyses", "tous les resultats", "tous les resultats",
    "l ensemble", "l'ensemble", "recapitulatif", "recapitulatif",
    "en resume", "en resume", "analyse complete", "analyse complete",
    "bilan complet", "tout le bilan", "interprete", "interprete",
    "analyse", "explique", "commente", "evalue", "evalue",
]


def _is_synthesis_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _SYNTHESIS_KEYWORDS)


# ?? Nodes LangGraph ???????????????????????????????????????????????????????????
def planning_node(state):
    state["iteration"] = state.get("iteration", 0) + 1
    state["needs_web"] = False
    state["web_used"] = state.get("web_used", False)
    return state


def generation_node(state):
    query = state["query"]
    docs  = state.get("retrieved_docs", [])
    context = "\n\n".join([d.get("text", "") for d in docs]) if docs else ""
    if any(d.get("is_ocr", False) for d in docs) and context:
        context = _clean_ocr_text(context)
    context = context[:4000] if context else ""
    context = _normalize_medical_context(context)

    extracted = extract_bilan_values(context) if context.strip() else []
    is_synthesis = _is_synthesis_query(query)

    if extracted:
        print("\n" + "=" * 60)
        print("[DEBUG] BILAN EXTRAIT (statuts Python) :")
        for r in extracted:
            unit_str = f" {r['unit']}" if r.get("unit") else ""
            print(f"  {r['label']}: {r['value']}{unit_str} -> {r['status']} (ref: {r['norm']})")
        print("=" * 60 + "\n")

        # CAS 3 : synthese deterministe, sans LLM
        if is_synthesis:
            state["answer"]    = _build_synthesis_response(extracted)
            state["needs_web"] = False
            return state

        # CAS 1 : question specifique sur les valeurs
        prompt = (
            "Extrait du bilan biologique du patient:\n"
            + "\n".join([
                r['label'] + ": " + str(r['value']) + " " + r.get('unit', '')
                + " statut " + r['status']
                for r in extracted
            ])
            + "\n\nQuestion: " + query
            + "\nReponse:"
        )

        try:
            response_text = llm.invoke(prompt)
            if not isinstance(response_text, str):
                response_text = str(response_text)

            print("\n" + "=" * 60)
            print("[DEBUG] REPONSE BRUTE LLM :")
            print(response_text[:600])
            print("=" * 60 + "\n")

            if "Reponse:" in response_text:
                response_text = response_text.split("Reponse:", 1)[-1].strip()

            is_valid, violations = validate_response(response_text, extracted)
            if not is_valid:
                print(f"[VALIDATION] {len(violations)} violation(s) detectee(s) :")
                for v in violations:
                    print(f"  X {v}")
                response_text = _append_validation_warning(response_text, violations)

            state["answer"]    = response_text
            state["needs_web"] = False

        except Exception as e:
            state["answer"] = f"Erreur LLM : {e}"

        return state

    # Pas de valeurs extraites
    if not context.strip():
        state["answer"]    = "INFORMATION ABSENTE"
        state["needs_web"] = not state.get("web_used", False)
        return state

    # CAS 2 : pas de valeurs structurees, on envoie le contexte brut
    prompt = (
        "Extrait du bilan biologique du patient:\n"
        + context
        + "\n\nQuestion: " + query
        + "\nReponse:"
    )

    try:
        response_text = llm.invoke(prompt)
        if not isinstance(response_text, str):
            response_text = str(response_text)

        print("\n" + "=" * 60)
        print("[DEBUG] REPONSE BRUTE BIOMISTRAL (fallback) :")
        print(repr(response_text))
        print("=" * 60 + "\n")

        if "Reponse:" in response_text:
            response_text = response_text.split("Reponse:", 1)[-1].strip()

        clean_lines   = [l.strip() for l in response_text.split("\n") if l.strip()]
        response_text = "\n".join(clean_lines)

        if context.strip() and len(response_text) > 80:
            ctx_start = context.strip()[:40].lower()
            if ctx_start in response_text.lower():
                response_text = "INFORMATION ABSENTE"

        bad_starts = ["voici les resultats", "voici les donnees", "bonjour"]
        if not response_text or any(response_text.lower().startswith(b) for b in bad_starts):
            response_text = "INFORMATION ABSENTE"

        state["answer"] = response_text

    except Exception as e:
        state["answer"] = f"Erreur LLM : {e}"
        return state

    # Detection refus -> declenchement de la recherche web
    refus_signals = [
        "information absente", "je ne sais pas", "je ne peux pas",
        "pas d'information", "pas trouve", "n'existe pas", "cannot find",
        "no information", "ne sont pas mentionne",
        "ne sont pas directement mentionne", "n'ont pas ete identifie",
        "ne sont pas specifiquement mentionne", "pas mentionne",
        "ne contient pas", "n'est pas mentionne", "ne sont pas precise",
        "n'apparait pas", "n'apparaissent pas",
    ]
    resp_lower = response_text.lower()
    is_refusal = (
        any(s in resp_lower for s in refus_signals)
        or len(response_text.strip()) < 5
    )

    keywords_map = {
        "plaquette":   ["plaquette", "plt"],
        "leucocyte":   ["leucocyte", "gb", "globules blancs"],
        "glucose":     ["glucose", "glycemie"],
        "cholesterol": ["cholesterol"],
    }
    query_lower   = query.lower()
    context_lower = context.lower()
    for key, variants in keywords_map.items():
        if key in query_lower and not any(v in context_lower for v in variants):
            is_refusal        = True
            state["answer"]   = "INFORMATION ABSENTE"
            break

    state["needs_web"] = is_refusal and not state.get("web_used", False)
    return state


def web_search_node(state):
    query       = state["query"]
    clean_query = re.sub(r"\b(quels?|quelles?|sont|les?|du|de|la|le)\b", "", query, flags=re.IGNORECASE)
    clean_query = re.sub(r"\s+", " ", clean_query).strip()
    search_query = clean_query + " medical symptomes"

    try:
        ddgs    = DDGS()
        results = list(ddgs.text(search_query, max_results=3))
        if results:
            parts = [
                "Source: " + r.get("title", "Sans titre") + "\nContenu: " + r.get("body", "")
                for r in results
            ]
            web_context = "\n\n".join(parts)
            state["retrieved_docs"] = [{"text": web_context, "source": "web", "is_web": True}]
        else:
            state["retrieved_docs"] = [
                {"text": "Aucun resultat trouve sur le web.", "source": "web", "is_web": True}
            ]
        state["web_used"]  = True
        state["needs_web"] = False
    except Exception as e:
        state["retrieved_docs"] = [
            {"text": f"Erreur recherche web : {e}", "source": "web", "is_web": True}
        ]
        state["web_used"]  = True
        state["needs_web"] = False

    return state
