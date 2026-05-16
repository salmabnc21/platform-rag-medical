 
import os
from dotenv import load_dotenv
from src.rag.query_engine import query_medical

load_dotenv()

def main():
    print('\n' + '='*60)
    print('PLATEFORME MÉDICALE INTELLIGENTE - RAG AGENTIC')
    print('='*60 + '\n')
    
    while True:
        question = input('\n Votre question médicale : ').strip()
        
        if question.lower() in ['quit', 'exit', 'quitter']:
            print('\n✓ Au revoir !')
            break
        
        if not question:
            print(' Veuillez poser une question.')
            continue
        
        print('\n Traitement en cours...')
        
        try:
            response = query_medical(question)
            print('\n' + '='*60)
            print(' RÉPONSE MÉDICALE :')
            print('='*60)
            print(f'\n{response}\n')
            print('='*60)
            
        except Exception as e:
            print(f'\n Erreur : {str(e)}')
            print('Vérifiez que Ollama est en train de tourner : ollama serve')

if __name__ == '__main__':
    main()