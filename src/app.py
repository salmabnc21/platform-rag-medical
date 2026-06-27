import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import PyPDF2
import docx
import io
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from src.agent.agent_graph import app as agent_app
from src.rag.query_engine import query_engine
from src.rag.temp_rag import build_temp_index, query_temp_index

st.set_page_config(
    page_title="Plateforme Médicale Intelligente",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .stApp { background-color: #F4FAF6; }
  section[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #D4E8DA; }

  .med-header {
    background: linear-gradient(135deg, #1B6B3A 0%, #2E9B57 100%);
    padding: 28px 32px; border-radius: 12px; margin-bottom: 28px;
    display: flex; align-items: center; gap: 18px;
  }
  .med-header-icon {
    width: 52px; height: 52px; background: rgba(255,255,255,0.15);
    border-radius: 12px; display: flex; align-items: center; justify-content: center;
  }
  .med-title { color: #FFFFFF; font-size: 22px; font-weight: 600; margin: 0; }
  .med-subtitle { color: rgba(255,255,255,0.8); font-size: 13px; margin-top: 4px; }

  .section-label {
    font-size: 12px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.07em; color: #4A8B60; margin-bottom: 10px;
    display: flex; align-items: center; gap: 8px;
  }

  .upload-box {
    background: #FFFFFF; border: 1.5px dashed #2E9B57;
    border-radius: 10px; padding: 20px; margin-bottom: 8px;
  }

  .file-types-row {
    display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap;
  }
  .file-type-badge {
    padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 500;
    border: 1px solid #D4E8DA; background: #FFFFFF; color: #4A8B60;
    display: inline-flex; align-items: center; gap: 5px;
  }
  .file-type-badge.active { background: #E8F5EC; border-color: #2E9B57; color: #1B6B3A; font-weight: 600; }

  .file-success {
    background: #F0FAF3; border: 1px solid #2E9B57; border-radius: 8px;
    padding: 12px 16px; margin-top: 10px; font-size: 13px; color: #1B6B3A;
  }
  .file-warning {
    background: #FFFBEB; border: 1px solid #F6C90E; border-radius: 8px;
    padding: 12px 16px; margin-top: 10px; font-size: 13px; color: #7B5E00;
  }

  .stTextArea textarea {
    border: 1.5px solid #D4E8DA !important; border-radius: 10px !important;
    background: #FFFFFF !important; font-size: 14px !important;
    color: #1B3A28 !important; padding: 14px !important;
  }
  .stTextArea textarea:focus {
    border-color: #2E9B57 !important;
    box-shadow: 0 0 0 3px rgba(46,155,87,0.12) !important;
  }

  .stButton > button {
    background-color: #1B6B3A !important; color: #FFFFFF !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 14px !important; padding: 10px 26px !important;
  }
  .stButton > button:hover { background-color: #155430 !important; }

  .answer-block {
    background: #FFFFFF; border: 1px solid #D4E8DA;
    border-left: 4px solid #2E9B57; border-radius: 10px;
    padding: 20px 22px; margin-top: 20px;
  }
  .answer-tag {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.06em; color: #2E9B57; margin-bottom: 10px;
  }
  .answer-text { font-size: 14.5px; line-height: 1.7; color: #1B3A28; }

  .stack-item {
    display: flex; align-items: center; gap: 8px; padding: 8px 10px;
    background: #F4FAF6; border: 1px solid #D4E8DA; border-radius: 8px;
    font-size: 13px; font-weight: 500; color: #1B3A28; margin-bottom: 8px;
  }
  .status-dot { width: 8px; height: 8px; background: #2E9B57; border-radius: 50%; display: inline-block; margin-right: 6px; }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 24px !important; }
</style>
""", unsafe_allow_html=True)

# ─── FONCTIONS D'EXTRACTION ───
def extract_pdf(file):
    reader = PyPDF2.PdfReader(file)
    pages = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            pages.append(t)
    return "\n".join(pages), len(reader.pages)

def extract_docx(file):
    doc = docx.Document(file)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs), len(paragraphs)

def extract_txt(file):
    content = file.read().decode("utf-8", errors="ignore")
    return content, content.count("\n")

def extract_image(file):
    try:
        image = Image.open(file)
        text = pytesseract.image_to_string(image, lang="fra+eng")
        if text.strip():
            return text, "OCR réussi"
        else:
            return "", "Aucun texte détecté dans l'image"
    except Exception as e:
        return "", f"OCR non disponible : {str(e)}"

# ─── HEADER ───
st.markdown("""
<div class="med-header">
  <div class="med-header-icon">
    <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.5" stroke-linecap="round">
      <circle cx="12" cy="12" r="10"/>
      <line x1="12" y1="8" x2="12" y2="16"/>
      <line x1="8" y1="12" x2="16" y2="12"/>
    </svg>
  </div>
  <div>
    <div class="med-title">Plateforme Médicale Intelligente</div>
    <div class="med-subtitle">Architecture Agentic RAG &nbsp;·&nbsp; BioMistral &nbsp;·&nbsp; ChromaDB &nbsp;·&nbsp; LangGraph</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR ───
with st.sidebar:
    st.markdown("""
    <div style="font-size:16px;font-weight:600;color:#1B3A28;margin-bottom:16px;">A propos</div>
    <div style="font-size:12px;color:#6B8E78;margin-bottom:20px;line-height:1.5;">
      Système de questions-réponses médicales basé sur une base documentaire indexée localement.
    </div>
    <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;color:#4A8B60;margin-bottom:10px;">Stack technique</div>
    <div class="stack-item">LangGraph</div>
    <div class="stack-item">LlamaIndex</div>
    <div class="stack-item">ChromaDB</div>
    <div class="stack-item">BioMistral 7B</div>
    <div style="margin-top:20px;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.07em;color:#4A8B60;margin-bottom:10px;">Mode de requête</div>
    """, unsafe_allow_html=True)

    mode = st.radio("", ["Agent + RAG", "RAG seul", "Agent seul"], label_visibility="collapsed")

    st.markdown("""
    <div style="margin-top:28px;padding-top:16px;border-top:1px solid #D4E8DA;">
      <div style="font-size:12px;color:#4A8B60;margin-bottom:8px;">
        <span class="status-dot"></span>Ollama actif — localhost:11434
      </div>
      <div style="font-size:11px;color:#6B8E78;">ENSA Oujda · MGSI · 2025–2026</div>
    </div>
    """, unsafe_allow_html=True)

# ─── COLONNES PRINCIPALES ───
col1, col2 = st.columns([3, 1])

with col1:

    # ─── TYPES DE FICHIERS SUPPORTÉS ───
    st.markdown("""
    <div class="section-label">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#4A8B60" stroke-width="2" stroke-linecap="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
      </svg>
      Importer un document (optionnel)
    </div>
    <div class="file-types-row">
      <span class="file-type-badge active">PDF</span>
      <span class="file-type-badge active">Word (.docx)</span>
      <span class="file-type-badge active">Image (JPG, PNG)</span>
      <span class="file-type-badge active">Texte (.txt)</span>
    </div>
    """, unsafe_allow_html=True)

    # ─── UPLOAD ───
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Glissez-déposez votre fichier ici ou cliquez pour parcourir",
        type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
        label_visibility="visible"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ─── TRAITEMENT DU FICHIER ───
    file_context = ""
    file_label = ""
    file_type_detected = ""

    if uploaded_file is not None:
        ext = uploaded_file.name.split(".")[-1].lower()
        try:
            if ext == "pdf":
                file_context, nb = extract_pdf(uploaded_file)
                file_type_detected = "PDF"
                info = f"{nb} page(s) · {len(file_context):,} caractères extraits"
                st.markdown(f'<div class="file-success"><strong>{uploaded_file.name}</strong> chargé · {info}</div>', unsafe_allow_html=True)

            elif ext == "docx":
                file_context, nb = extract_docx(uploaded_file)
                file_type_detected = "Word"
                info = f"{nb} paragraphe(s) · {len(file_context):,} caractères extraits"
                st.markdown(f'<div class="file-success"><strong>{uploaded_file.name}</strong> chargé · {info}</div>', unsafe_allow_html=True)

            elif ext == "txt":
                file_context, nb = extract_txt(uploaded_file)
                file_type_detected = "Texte"
                info = f"{nb} lignes · {len(file_context):,} caractères"
                st.markdown(f'<div class="file-success"><strong>{uploaded_file.name}</strong> chargé · {info}</div>', unsafe_allow_html=True)

            elif ext in ["jpg", "jpeg", "png"]:
                file_type_detected = "Image"
                image = Image.open(uploaded_file)
                st.image(image, caption=uploaded_file.name, use_column_width=True)
                file_context, ocr_status = extract_image(uploaded_file)
                if file_context.strip():
                    info = f"{len(file_context):,} caractères extraits par OCR"
                    st.markdown(f'<div class="file-success"><strong>{uploaded_file.name}</strong> · {info}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="file-warning"><strong>Attention :</strong> {ocr_status}. L\'image sera tout de même transmise comme référence.</div>', unsafe_allow_html=True)
                with st.expander("Voir le texte extrait (debug)"):
                    st.text(file_context)

            file_label = f"{file_type_detected} uploadé : {uploaded_file.name}"

            # Build temporary RAG index — only when the file changes
            file_key = uploaded_file.name
            if file_context.strip() and st.session_state.get("temp_rag_file") != file_key:
                with st.spinner("Indexation du document (RAG temporaire)…"):
                    try:
                        client, temp_idx, n_chunks = build_temp_index(file_context)
                        st.session_state["temp_rag_client"] = client
                        st.session_state["temp_rag_index"] = temp_idx
                        st.session_state["temp_rag_file"] = file_key
                        st.markdown(
                            f'<div class="file-success">RAG temporaire prêt · {n_chunks} chunks indexés</div>',
                            unsafe_allow_html=True,
                        )
                    except Exception as ie:
                        st.warning(f"Indexation RAG échouée ({ie}) — contexte brut utilisé en fallback.")

        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier : {str(e)}")

    # ─── QUESTION ───
    st.markdown("""
    <div class="section-label" style="margin-top:24px;">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#4A8B60" stroke-width="2" stroke-linecap="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
      </svg>
      Question médicale
    </div>
    """, unsafe_allow_html=True)

    question = st.text_area(
        "",
        placeholder="Ex : Quels sont les traitements recommandés pour l'hypertension chez un patient diabétique ?",
        height=120,
        label_visibility="collapsed"
    )

    col_btn, col_mode_info = st.columns([1, 3])
    with col_btn:
        submit = st.button("Soumettre la requête", use_container_width=True)
    with col_mode_info:
        src = file_label if uploaded_file else "Base documentaire indexée"
        st.markdown(f"""
        <div style="padding-top:8px;font-size:12px;color:#6B8E78;">
          Mode : <strong style="color:#1B6B3A;">{mode}</strong> &nbsp;·&nbsp;
          Source : <strong style="color:#1B6B3A;">{src}</strong>
        </div>
        """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background:#FFFFFF;border:1px solid #D4E8DA;border-radius:10px;padding:16px;margin-bottom:14px;">
      <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;color:#4A8B60;margin-bottom:12px;">Base documentaire</div>
      <div style="font-size:24px;font-weight:600;color:#1B3A28;">5</div>
      <div style="font-size:12px;color:#6B8E78;margin-bottom:12px;">PDFs indexés</div>
      <div style="font-size:24px;font-weight:600;color:#1B3A28;">11 242</div>
      <div style="font-size:12px;color:#6B8E78;">chunks vectorisés</div>
    </div>
    <div style="background:#FFFFFF;border:1px solid #D4E8DA;border-radius:10px;padding:16px;">
      <div style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;color:#4A8B60;margin-bottom:12px;">Formats supportés</div>
      <div style="font-size:13px;color:#1B3A28;line-height:1.8;">
        PDF · Word<br>Image (OCR)<br>Texte brut
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─── TRAITEMENT ───
if submit:
    if not question.strip():
        st.warning("Veuillez saisir une question avant de soumettre.")
    else:
        with st.spinner("Traitement en cours..."):
            try:
                if file_context:
                    is_ocr = (file_type_detected == "Image")
                    if "temp_rag_index" in st.session_state:
                        relevant_chunks = query_temp_index(
                            st.session_state["temp_rag_index"],
                            question,
                            top_k=3,
                        )
                        if not relevant_chunks.strip():
                            relevant_chunks = file_context[:4000]
                    else:
                        relevant_chunks = file_context[:4000]

                    result = agent_app.invoke({
                        'query': question,
                        'retrieved_docs': [{'text': relevant_chunks, 'is_ocr': is_ocr}],
                        'answer': '',
                        'iteration': 0,
                        'needs_web': False,
                        'memory_context': ''
                    })
                    answer = result['answer']
                    source_label = file_label

                elif mode == "Agent + RAG":
                    retrieved = query_engine.query(question)
                    result = agent_app.invoke({
                        'query': question,
                        'retrieved_docs': [{'text': retrieved.response}] if retrieved.response else [],
                        'answer': '',
                        'iteration': 0,
                        'needs_web': False,
                        'memory_context': ''
                    })
                    answer = result['answer']
                    source_label = "Agent + RAG — Base documentaire indexée"

                elif mode == "RAG seul":
                    response = query_engine.query(question)
                    answer = response.response if response.response else "Aucun document pertinent trouvé dans la base."
                    source_label = "RAG — Base documentaire indexée"

                else:
                    result = agent_app.invoke({
                        'query': question,
                        'retrieved_docs': [],
                        'answer': '',
                        'iteration': 0,
                        'needs_web': False,
                        'memory_context': ''
                    })
                    answer = result['answer']
                    source_label = "Agent seul — Raisonnement BioMistral"

                st.write("DEBUG - web_used:", result.get("web_used"))
                st.write("DEBUG - needs_web:", result.get("needs_web"))
                st.write("DEBUG - sources:", result.get("retrieved_docs"))

                st.markdown(f"""
                <div class="answer-block">
                  <div class="answer-tag">Réponse générée &nbsp;·&nbsp; {source_label}</div>
                  <div class="answer-text">{answer}</div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.markdown(f"""
                <div style="background:#FFF5F5;border:1px solid #FFC5C5;border-left:4px solid #E53E3E;
                  border-radius:10px;padding:16px 20px;margin-top:16px;">
                  <div style="font-size:12px;font-weight:600;color:#C53030;text-transform:uppercase;
                    letter-spacing:0.05em;margin-bottom:8px;">Erreur</div>
                  <div style="font-size:13px;color:#742A2A;">{str(e)}</div>
                  <div style="font-size:12px;color:#9B6B6B;margin-top:8px;">
                    Vérifiez qu'Ollama est actif : <code>ollama serve</code>
                  </div>
                </div>
                """, unsafe_allow_html=True)

# ─── PIED DE PAGE ───
st.markdown("""
<div style="margin-top:48px;padding-top:16px;border-top:1px solid #D4E8DA;
  display:flex;justify-content:space-between;font-size:12px;color:#6B8E78;">
  <span>Plateforme Médicale Intelligente — ENSA Oujda</span>
  <span>Données traitées localement · Aucune transmission externe</span>
</div>
""", unsafe_allow_html=True)
