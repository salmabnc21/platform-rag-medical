from src.rag.query_engine import query_medical
from src.graph.ner_extractor import extract_medical_entities

if __name__ == "__main__":
    # Test RAG
    response = query_medical("Quels sont les symptômes du diabète ?")
    print("RAG OK:", response[:100])
    
    # Test NER
    text = "Le patient souffre de diabète et prend de la metformine."
    entities = extract_medical_entities(text)
    print("NER OK:", entities)