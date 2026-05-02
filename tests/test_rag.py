from src.rag.query_engine import query_medical
from src.graph.ner_extractor import extract_medical_entities

def test_query():
    response = query_medical("Quels sont les symptômes du diabète ?")
    assert response is not None
    assert len(response) > 0
    print("✅ Query OK:", response[:100])

def test_ner():
    text = "Le patient souffre de diabète et prend de la metformine."
    entities = extract_medical_entities(text)
    assert isinstance(entities, dict)
    print("✅ NER OK:", entities)

if __name__ == "__main__":
    test_query()
    test_ner()