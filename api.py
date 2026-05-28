from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from scipy.stats import mode
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")

app = FastAPI(
    title="API de Segmentação CRM",
    description="API RESTful para integração do modelo de Machine Learning com o back-end/front-end do site real.",
    version="1.0.0"
)

# Configuração de CORS para permitir que o Front-end comunique com esta API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção coloca-se o domínio exato do site
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Definição do formato de entrada (Contrato da API com o Site)
class DadosCliente(BaseModel):
    Renda: float = 50000.0
    Gasto_Total: float = 1500.0
    Gasto_Medio_Compra: float = 150.0
    Total_Compras: int = 10
    Compras_Web: int = 3
    Compras_Loja: int = 5
    Compras_Catalogo: int = 1
    Compras_Com_Desconto: int = 2
    Gasto_Vinhos: float = 500.0
    Gasto_Frutas: float = 50.0
    Gasto_Carnes: float = 300.0
    Gasto_Peixes: float = 80.0
    Gasto_Doces: float = 40.0
    Gasto_Ouro: float = 100.0

# Funções auxiliares mantidas do projeto original
def alinhar_labels(ref, target):
    return 1 - target if accuracy_score(ref, target) < 0.5 else target

class ConsensusModel:
    def __init__(self, km, hi, gm):
        self.km = km
        self.hi = hi
        self.gm = gm
    
    def predict(self, X):
        if len(X) < 5:
            return self.km.predict(X)
        l_km = self.km.predict(X)
        l_hi = self.hi.fit_predict(X)
        l_gm = self.gm.predict(X)
        l_hi_aln = alinhar_labels(l_km, l_hi)
        l_gm_aln = alinhar_labels(l_km, l_gm)
        votos = np.vstack((l_km, l_hi_aln, l_gm_aln))
        labels, _ = mode(votos, axis=0)
        return labels.ravel()

# Workaround para carregar os modelos exportados através do Uvicorn
import __main__
__main__.ConsensusModel = ConsensusModel

def select_features_for_artifact(artifact, df: pd.DataFrame) -> pd.DataFrame:
    if hasattr(artifact, "feature_names_in_"):
        expected = list(artifact.feature_names_in_)
        return df.reindex(columns=expected, fill_value=0)
    return df

def predict_cluster(df: pd.DataFrame, scaler, model) -> int:
    model_input = select_features_for_artifact(scaler, df)
    scaled = scaler.transform(model_input)
    pred = model.predict(scaled)
    if isinstance(pred, (list, np.ndarray, pd.Series)):
        return int(pred[0])
    return int(pred)

# Carregamento dos modelos
models = {}
@app.on_event("startup")
def load_models():
    global models
    base_path = Path(__file__).parent / "modelos_exportados"
    try:
        models = {
            "kmeans_geral_scaler": joblib.load(base_path / "kmeans_geral_scaler.pkl"),
            "kmeans_geral_model": joblib.load(base_path / "kmeans_geral_model.pkl"),
            "consenso_canais_scaler": joblib.load(base_path / "consenso_canais_scaler.pkl"),
            "consenso_canais_model": joblib.load(base_path / "consenso_canais_model.pkl"),
            "consenso_produtos_scaler": joblib.load(base_path / "consenso_produtos_scaler.pkl"),
            "consenso_produtos_model": joblib.load(base_path / "consenso_produtos_model.pkl"),
        }
    except Exception as e:
        print(f"Erro ao carregar modelos: {e}")

# 2. Rota Principal da API (A que o site vai chamar)
@app.post("/api/v1/segmentar")
def segmentar_cliente(cliente: DadosCliente):
    try:
        # Converte os dados recebidos pelo site (JSON) para um DataFrame Pandas
        dados_df = pd.DataFrame([cliente.dict()])
        
        # Faz as predições
        cluster_geral = predict_cluster(dados_df, models["kmeans_geral_scaler"], models["kmeans_geral_model"])
        cluster_canais = predict_cluster(dados_df, models["consenso_canais_scaler"], models["consenso_canais_model"])
        cluster_produtos = predict_cluster(dados_df, models["consenso_produtos_scaler"], models["consenso_produtos_model"])
        
        # Motor de Regras ajustado baseado na análise real dos centróides
        if cluster_geral == 1:
            acao = "VIP / Premium: Acionar contato personalizado com curadoria avançada. Alto LTV (Customer Lifetime Value)."
        else:
            acao = "Retenção / Descontos: Público de entrada com menor renda. Focar em ofertas para aumentar frequência de compra."

        msg_produtos = {
            0: "Alto consumo premium (Vinhos e Carnes). Focar em cross-sell de ticket alto.",
            1: "Gasto contido. Sugerir combo de essenciais e itens de baixo ticket."
        }
        
        msg_canais = {
            0: "Comprador Omnichannel agressivo (Loja Física forte e Catálogo). Estratégia multicanal completa.",
            1: "Comprador esporádico. Focar em cupons de desconto por mídias digitais para engajamento."
        }

        # Retorna o JSON para o site renderizar as coisas ou salvar no banco de dados
        return {
            "status": "sucesso",
            "clusters": {
                "geral": cluster_geral,
                "canais": cluster_canais,
                "produtos": cluster_produtos
            },
            "recomendacao_produtos": msg_produtos.get(cluster_produtos, "Sem recomendação de produtos."),
            "recomendacao_canais": msg_canais.get(cluster_canais, "Sem recomendação de canais."),
            "acao_marketing": acao
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
