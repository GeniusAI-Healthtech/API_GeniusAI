import os
from dotenv import load_dotenv
from openai import OpenAI

# Carrega as variáveis de ambiente
load_dotenv()

# Recupera a chave da API de um arquivo .env
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("A chave da API OpenAI não foi encontrada. Verifique seu arquivo .env.")

# Inicializa o cliente com a chave da API
client = OpenAI(api_key=api_key)
