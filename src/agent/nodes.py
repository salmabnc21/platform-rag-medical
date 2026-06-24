from langchain_community.llms import Ollama
from duckduckgo_search import DDGS
import re

llm = Ollama(
    model="cniongolo/biomistral",
    base_url="http://localhost:11434",
    temperature=0.0,
    top_p=0.9,
    top_k=40,
)

# Abréviations médicales courantes → formes complètes pour aider le LLM
_ABBREV = [
    (r'\bHb\b',   'Hémoglobine (Hb)'),
    (r'\bHGB\b',  'Hémoglobine (HGB)'),
    (r'\bGR\b',   'Hématies / Globules rouges (GR)'),
    (r'\bRBC\b',  'Hématies / Érythrocytes (RBC)'),
    (r'\bGB\b',   'Leucocytes / Globules blancs (GB)'),
    (r'\bWBC\b',  'Leucocytes (WBC)'),
    (r'\bPLT\b',  'Plaquettes (PLT)'),
    (r'\bHCT\b',  'Hématocrite (HCT)'),
    (r'\bVGM\b',  'Volume globulaire moyen (VGM)'),
    (r'\bMCV\b',  'Volume corpusculaire moyen (MCV)'),
    (r'\bTCMH\b', 'Teneur corpusculaire en hémoglobine (TCMH)'),
    (r'\bMCH\b',  'Teneur corpusculaire en hémoglobine (MCH)'),
    (r'\bCCMH\b', 'Concentration corpusculaire en hémoglobine (CCMH)'),
    (r'\bMCHC\b', 'Concentration corpusculaire en hémoglobine (MCHC)'),
    (r'\bIDR\b',  'Indice de distribution des réticulocytes (IDR)'),
    (r'\bRDW\b',  'Distribution des globules rouges (RDW)'),
    (r'\bVS\b',   'Vitesse de sédimentation (VS)'),
    (r'\bCRP\b',  'Protéine C-réactive (CRP)'),
    (r'\bNFS\b',  'Numération formule sanguine (NFS)'),
    (r'\bTP\b',   'Taux de prothrombine (TP)'),
    (r'\bINR\b',  'Rapport normalisé international (INR)'),
    (r'\bTCA\b',  'Temps de céphaline activée (TCA)'),
    (r'\bGly\b',  'Glycémie (Gly)'),
    (r'\bHbA1c\b','Hémoglobine glyquée (HbA1c)'),
]

def _normalize_medical_context(text: str) -> str:
    """Remplace les abréviations médicales par leurs formes longues pour aider le LLM."""
    for pattern, replacement in _ABBREV:
        text = re.sub(pattern, replacement, text)
    return text


# ── Extraction déterministe des valeurs de bilan ─────────────────────────────
_RE_NORM_RANGE = re.compile(r'N[z:]?\s*([\d,. ]+?)\s*[-–]\s*([\d,.]+)', re.IGNORECASE)
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
        return f"{low} – {high}"
    if nt == 'lt':
        return f"< {high}"
    if nt == 'gt':
        return f"> {low}"
    return "?"


def _bilan_status(value: float, norm) -> str:
    if norm is None:
        return "non évalué"
    low, high, nt = norm
    cmp = value
    ref = high if high is not None else low
    if ref and ref > 0:
        ratio = value / ref
        if ratio > 500_000:
            cmp = value / 1_000_000
        elif ratio > 500:
            cmp = value / 1_000
    if nt == 'range':
        if cmp < low:  return "bas"
        if cmp > high: return "élevé"
        return "normal"
    if nt == 'lt':
        return "normal" if cmp < high else "élevé"
    if nt == 'gt':
        return "normal" if cmp > low  else "bas"
    return "non évalué"


def _parse_bilan_line(line: str):
    line = line.strip()
    if not line:
        return None
    nm = _RE_NORM_START.search(line)
    norm_part = nm.group(1).strip() if nm else ''
    data_part  = line[:nm.start()].strip() if nm else line
    vm = _RE_NUM.search(data_part)
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
        "label":    label,
        "value":    value,
        "unit":     unit,
        "norm":     _bilan_norm_str(norm),
        "status":   _bilan_status(value, norm),
    }


def extract_bilan_values(ocr_text: str) -> list:
    """Extrait les triplets (label, valeur, norme) d'un texte OCR de bilan. Sans LLM."""
    return [r for r in (_parse_bilan_line(l) for l in ocr_text.splitlines()) if r]


def _build_synthesis_response(rows: list) -> str:
    """Construit la réponse de synthèse en Python pur à partir des valeurs extraites."""
    lines = ["Voici toutes les valeurs relevées dans ce bilan :\n"]
    for r in rows:
        unit_str  = f" {r['unit']}" if r['unit'] else ""
        norm_str  = f" (réf : {r['norm']})" if r['norm'] != "sans norme" else ""
        status    = r['status']
        lines.append(f"• {r['label']} : {r['value']}{unit_str} → {status}{norm_str}")
    return "\n".join(lines)


_SYNTHESIS_KEYWORDS = [
    "résume", "resume", "synthèse", "synthese", "ensemble",
    "toutes les valeurs", "vue d ensemble", "vue d'ensemble",
    "toutes les analyses", "tous les résultats", "tous les resultats",
    "l ensemble", "l'ensemble", "récapitulatif", "recapitulatif",
    "en résumé", "en resume", "analyse complète", "analyse complete",
    "bilan complet", "tout le bilan",
]

def _is_synthesis_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _SYNTHESIS_KEYWORDS)


def planning_node(state):
    state["iteration"] = state.get("iteration", 0) + 1
    state["needs_web"] = False
    state["web_used"] = state.get("web_used", False)
    return state


def generation_node(state):
    query = state["query"]
    docs = state.get("retrieved_docs", [])
    context = "\n".join([d.get("text", "") for d in docs]) if docs else ""
    context = context[:3000] if context else ""
    context = _normalize_medical_context(context)

    is_synthesis = _is_synthesis_query(query)

    if context.strip():
        if is_synthesis:
            # Extraction déterministe : pas de LLM pour lire les chiffres
            extracted = extract_bilan_values(context)
            if extracted:
                state["answer"] = _build_synthesis_response(extracted)
                state["needs_web"] = False
                return state
            # Fallback si aucune valeur structurée détectée
            prompt = (
                "Tu es un assistant médical. Voici l'extrait d'un bilan biologique:\n\n"
                + context
                + "\n\nTâche: Liste uniquement les paramètres et valeurs présents dans"
                " ce texte, sans en inventer. Pour chaque ligne, recopie le nom, la"
                " valeur et l'unité exactement tels qu'ils apparaissent.\n\nRésumé:"
            )
        else:
            prompt = (
                "Extrait du bilan biologique du patient:\n"
                + context
                + "\n\nQuestion: " + query
                + "\nReponse en une phrase courte avec la valeur numerique. "
                "Si la valeur n'est pas dans le bilan: INFORMATION ABSENTE\nReponse:"
            )
    else:
        prompt = "Question: " + query + "\nReponse: INFORMATION ABSENTE"

    try:
        response = llm.invoke(prompt)
        response_text = response if isinstance(response, str) else str(response)

        print("\n" + "="*60)
        print("[DEBUG] REPONSE BRUTE BIOMISTRAL:")
        print(repr(response_text))
        print("="*60 + "\n")

        for marker in ["Résumé:", "Reponse courte:", "Reponse:"]:
            if marker in response_text:
                response_text = response_text.split(marker, 1)[-1].strip()

        lines = response_text.split("\n")
        clean_lines = [line.strip() for line in lines if line.strip()]
        response_text = "\n".join(clean_lines)

        # Pour les questions non-synthèse : détecter si le modèle a écho le contexte
        if not is_synthesis and context.strip() and len(response_text) > 80:
            ctx_start = context.strip()[:40].lower()
            if ctx_start in response_text.lower():
                response_text = "INFORMATION ABSENTE"

        bad_starts = ["voici les resultats", "voici les donnees", "bonjour"]
        if not response_text or any(response_text.lower().startswith(b) for b in bad_starts):
            response_text = "INFORMATION ABSENTE"

        state["answer"] = response_text
    except Exception as e:
        state["answer"] = "Erreur LLM: " + str(e)
        return state

    # Les questions de synthèse ne déclenchent jamais la recherche web
    if is_synthesis:
        state["needs_web"] = False
        return state

    refus_signals = [
        "information absente",
        "je ne sais pas", "je ne peux pas",
        "pas d'information", "pas trouve", "n'existe pas", "cannot find",
        "no information", "ne sont pas mentionne", "ne sont pas directement mentionne",
        "n'ont pas ete identifie", "ne sont pas specifiquement mentionne",
        "pas mentionne", "ne contient pas", "n'est pas mentionne", "ne sont pas precise",
        "n'apparait pas", "n'apparaissent pas",
    ]

    response_lower = response_text.lower()
    is_refusal = False
    for signal in refus_signals:
        if signal in response_lower:
            is_refusal = True
            break
    if len(response_text.strip()) < 5:
        is_refusal = True

    keywords_map = {
        "plaquette": ["plaquette", "plt"],
        "leucocyte": ["leucocyte", "gb", "globules blancs"],
        "glucose": ["glucose", "glycemie"],
        "cholesterol": ["cholesterol"],
    }
    query_lower = query.lower()
    context_lower = context.lower()
    for key, variants in keywords_map.items():
        if key in query_lower:
            if not any(v in context_lower for v in variants):
                is_refusal = True
                state["answer"] = "INFORMATION ABSENTE"
                break

    if is_refusal and not state.get("web_used", False):
        state["needs_web"] = True
    else:
        state["needs_web"] = False

    return state


def web_search_node(state):
    query = state["query"]
    clean_query = re.sub(r"\b(quels?|quelles?|sont|les?|du|de|la|le)\b", "", query, flags=re.IGNORECASE)
    clean_query = re.sub(r"\s+", " ", clean_query).strip()
    search_query = clean_query + " medical symptomes"

    try:
        ddgs = DDGS()
        results = list(ddgs.text(search_query, max_results=3))

        if results:
            parts = []
            for r in results:
                title = r.get("title", "Sans titre")
                body = r.get("body", "")
                parts.append("Source: " + title + "\nContenu: " + body)
            web_context = "\n\n".join(parts)
            state["retrieved_docs"] = [{"text": web_context, "source": "web", "is_web": True}]
        else:
            state["retrieved_docs"] = [{"text": "Aucun resultat trouve sur le web.", "source": "web", "is_web": True}]

        state["web_used"] = True
        state["needs_web"] = False

    except Exception as e:
        state["retrieved_docs"] = [{"text": "Erreur recherche web: " + str(e), "source": "web", "is_web": True}]
        state["web_used"] = True
        state["needs_web"] = False

    return state
