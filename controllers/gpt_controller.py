from fastapi import APIRouter, HTTPException
from openai import OpenAI
import os
from dotenv import load_dotenv

router = APIRouter()

# Carrega as variáveis de ambiente
load_dotenv()

# Recupera a chave da API de um arquivo .env
api_key = os.getenv("OPENAI_API_KEY")

# Inicializa o cliente com a chave da API
client = OpenAI(api_key=api_key)

@router.post("/chat", tags=["ChatGPT"], summary="Converse com ChatGPT")
async def chat_with_gpt(detection_results: dict):
    """
    Recebe um prompt de texto e retorna uma resposta do modelo GPT-4 Turbo.

    Args:
        detection_results (dict): JSON object containing detection results.

    Returns:
        dict: Resposta do ChatGPT em formato JSON.
    """
    prompt = (
        "descreva em um único parágrafo simples a interpretação clínica dos dados de ECG fornecidos, "
        "Faça isso em português. "
        "o retorno deve ser um unico paragrafo"
        "Dados de ECG: " + str(detection_results)
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "Please output the response as a single, simple paragraph in Portuguese."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,  # Menor temperatura para maior coerência e precisão
            #max_tokens=150  # Reduzindo o número de tokens para limitar a extensão da resposta
        )

        message_content = response.choices[0].message.content
        #print('resposta GPT:', message_content)
        return {"response": message_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to communicate with ChatGPT API: {str(e)}")
