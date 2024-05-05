```
  uvicorn main:app --reload

```

## Modelo Organizacional 
📁 /project_root
│
├── 📁 app/
│   ├── __init__.py
│   └── main.py          # Arquivo principal que inicializa e configura o aplicativo FastAPI
│
├── 📁 controllers/
│   ├── object_detection_controller.py  # Controller para detecção de objetos
│   ├── ecg_controller.py               # Controller específico para operações de ECG
│   └── healthcheck_controller.py       # Controller para healthcheck da API
│
├── 📁 services/
│   ├── image_processing.py             # Serviços relacionados ao processamento de imagens
│   └── model_services.py               # Serviços relacionados ao gerenciamento de modelos
│
├── 📁 models/
│   └── image_models.py                 # Modelos Pydantic para dados de imagens
│
├── 📁 utils/
│   ├── logger.py                       # Utilitários para configuração de logging
│   └── image_utils.py                  # Utilitários para manipulação de imagens
│
├── requirements.txt                    # Arquivo para gerenciamento de dependências
└── .env                                # Arquivo para variáveis de ambiente



## Logs

from loguru import logger

logger.info("Modelo carregado: {}", model_sample_model.model_name)

## lint

Execute Flake8: Abra o terminal no diretório do seu projeto e execute o comando:
bash
Copy code
flake8 path/to/your/file.py
Substitua path/to/your/file.py pelo caminho do arquivo que deseja verificar. Se desejar verificar todo o projeto, pode simplesmente rodar flake8 ..
Verifique os Resultados: Flake8 listará todos os problemas encontrados, incluindo "E501 line too long (x > 79 characters)", que é o erro padrão para linhas que excedem o limite padrão de comprimento (79 caracteres por padrão).
Usando Black
Black vai formatar seu código automaticamente para um estilo consistente, que inclui quebrar linhas longas conforme necessário.

Execute Black no Modo de Checagem: Para verificar se o arquivo está de acordo com o estilo do Black sem alterar o arquivo, você pode usar:
bash
Copy code
black --check path/to/your/file.py
Este comando só verifica se o código segue o estilo do Black.
Execute Black para Formatar: Se quiser que o Black formate seu código automaticamente, execute:
bash
Copy code
black path/to/your/file.py
Isso reformatará seu código, incluindo a quebra de linhas longas para se ajustarem ao padrão do Black (88 caracteres por padrão).