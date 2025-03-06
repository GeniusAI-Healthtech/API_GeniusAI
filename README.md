# Documentação da API

Esta documentação fornece informações essenciais para executar, entender e trabalhar com a API. Inclui instruções para iniciar o projeto, descrição da estrutura organizacional do código, configuração de logs e detalhes sobre formatação e verificação de estilo de código.

## Iniciando o Projeto

Para rodar o projeto localmente, utilize o seguinte comando:

```bash
uvicorn main:app --reload
```

Este comando inicia o servidor de desenvolvimento com hot reload, permitindo atualizações automáticas sempre que o código for modificado.

## Estrutura do Projeto

A organização do código está detalhada abaixo:

```
📁 /project_root
│
├── main.py  # Arquivo principal que inicializa e configura o aplicativo FastAPI
│        
├── 📁 AI/  # Agrupa os modelos de IA por áreas.
│   ├── 📁 cardiac
│        
├── 📁 controllers/
│   ├── detection_controller.py  # Controller para detecção de objetos
│   └── healthcheck_controller.py  # Controller para healthcheck da API
│
├── 📁 services/
│   ├── image_processing.py  # Serviços relacionados ao processamento de imagens
│   └── model_services.py  # Serviços relacionados ao gerenciamento de modelos
│
├── 📁 models/
│   └── EX: Prisma  # Camada para manipulação e persistencia dos dados (Não Implementado)
│
├── 📁 utils/
│   ├── logger.py  # Utilitários para configuração de logging
│   └── image_utils.py  # Utilitários para manipulação de imagens
│
├── requirements.txt  # Arquivo para gerenciamento de dependências
└── .env  # Arquivo para variáveis de ambiente
```

## Logging

Configuramos o logging utilizando a biblioteca `loguru`, simplificando o monitoramento e a depuração da aplicação. A configuração do logger pode ser encontrada em `utils/logger.

Para usar o logger em outros arquivos:

```python
from utils.logger import get_logger
logger = get_logger()
```

## Lint e Formatação de Código

### Flake8

Para verificar o estilo e erros no código:

```bash
flake8 path/to/your/file.py
```

Substitua `path/to/your/file.py` pelo caminho desejado ou use `flake8 .` para verificar todo o projeto.

### Black

Para formatar o código automaticamente:

```bash
black --check path/to/your/file.py
```

Para verificar se o código segue o estilo do Black sem alterar os arquivos, ou:

```bash
black path/to/your/file.py
```

Para formatar o código automaticamente, incluindo a quebra de linhas longas.

---
