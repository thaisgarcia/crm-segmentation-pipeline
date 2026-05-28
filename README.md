# 🎯 Sistema de Segmentação de Clientes (CRM)

Este projeto contém um sistema de Machine Learning para segmentação de clientes, disponibilizado em duas abordagens diferentes e complementares. O objetivo é demonstrar tanto a criação de um MVP/Dashboard interativo quanto uma arquitetura voltada para Produção (API REST).

## 🏗️ Duas Abordagens Disponíveis

Ao trabalhar com Machine Learning em ambientes reais, passamos por duas fases importantes: a Prova de Conceito (PoC) e a Produtização (Deploy). Este projeto exemplifica ambas:

### 1. Aplicação Streamlit (`app.py`) - Dashboard & MVP
- **Objetivo:** Interface rápida e interativa para Cientistas de Dados, Analistas de Negócios e stakeholders testarem o modelo.
- **Vantagem:** Desenvolvimento ágil feito 100% em Python. Excelente para validar regras de negócios e explorar predições visualmente com equipes não técnicas.

### 2. FastAPI + Front-end (`api.py` + `index.html`) - Arquitetura de Produção
- **Objetivo:** Simular um ambiente real onde o modelo de ML atua como um microsserviço (Back-end) acessado remotamente por uma interface web (Front-end).
- **Vantagem:**
  - O `api.py` cria rotas RESTful, permitindo que múltiplos sistemas diferentes (site corporativo, app mobile, CRM interno) consumam as predições via JSON de forma escalável.
  - O `index.html` encapsula a visão do usuário final, fazendo integrações via JavaScript.

---

## 🚀 Como Executar o Projeto

Primeiro, instale todas as dependências necessárias para as duas abordagens:
```bash
pip install -r requirements.txt
```

### ▶️ Opção 1: Rodando via Streamlit (Dashboard)
Para visualizar a interface analítica interativa de forma instantânea:
```bash
streamlit run app.py
```
Acesse no seu navegador em: `http://localhost:8501`

### ▶️ Opção 2: Rodando via FastAPI (API + Site)
Para subir o microsserviço de Machine Learning simulando um ambiente de produção:

1. **Inicie a API (Back-end):**
```bash
uvicorn api:app --reload
```
A API estará disponível em `http://localhost:8000`. Você pode testar e validar os endpoints pela documentação interativa baseada no Swagger através de: `http://localhost:8000/docs`.

2. **Abra o Front-end (Site):**
Dê dois cliques no arquivo `index.html` para abri-lo diretamente no navegador, ou utilize extensões como "Live Server" no VS Code. O formulário na interface HTML cuidará de se comunicar em tempo real com a API rodando no terminal.
