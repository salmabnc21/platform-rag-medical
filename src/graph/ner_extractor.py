import spacy

nlp = spacy.load('en_core_web_sm')

def extract_medical_entities(text: str) -> dict:
    doc = nlp(text)
    entities = {
        'maladies':    [],
        'medicaments': [],
        'symptomes':   [],
        'personnes':   [],
    }
    for ent in doc.ents:
        if ent.label_ in ('PER',):
            entities['personnes'].append(ent.text)
        elif ent.label_ in ('MISC',):
            entities['medicaments'].append(ent.text)
    return entities