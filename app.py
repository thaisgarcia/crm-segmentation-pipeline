import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from scipy.stats import mode
from sklearn.metrics import accuracy_score
import warnings

# Ignora os avisos de versão do scikit-learn ao carregar os modelos antigos
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Trying to unpickle estimator.*")

def alinhar_labels(ref, target):
    return 1 - target if accuracy_score(ref, target) < 0.5 else target

class ConsensusModel:
    def __init__(self, km, hi, gm):
        self.km = km
        self.hi = hi
        self.gm = gm
    
    def predict(self, X):
        # O AgglomerativeClustering não suporta predição de novas amostras soltas e 
        # requer no mínimo 2 amostras. Além disso, 'alinhar_labels' depende de um lote 
        # grande para calcular 'accuracy_score' com precisão.
        # Portanto, para inferência em produção (1 cliente por vez), priorizamos o KMeans:
        if len(X) < 5:
            return self.km.predict(X)
            
        l_km = self.km.predict(X)
        l_hi = self.hi.fit_predict(X) # Agglomerative não tem predict separado
        l_gm = self.gm.predict(X)
        l_hi_aln = alinhar_labels(l_km, l_hi)
        l_gm_aln = alinhar_labels(l_km, l_gm)
        votos = np.vstack((l_km, l_hi_aln, l_gm_aln))
        labels, _ = mode(votos, axis=0)
        return labels.ravel()

# 1. Configuração da Página
st.set_page_config(page_title="CRM Inteligente - Segmentação", page_icon="🎯", layout="wide")
st.title("🎯 Sistema de Segmentação de Clientes (CRM)")
st.markdown("Insira os dados do cliente para prever os clusters de **Perfil Geral**, **Canais de Compra** e **Tipos de Produtos**.")

# 2. Carregar Modelos (usando cache para não carregar toda hora)
@st.cache_resource
def load_models():
    base_path = Path(__file__).parent / "modelos_exportados"
    return {
        "kmeans_geral_scaler": joblib.load(base_path / "kmeans_geral_scaler.pkl"),
        "kmeans_geral_model": joblib.load(base_path / "kmeans_geral_model.pkl"),
        "consenso_canais_scaler": joblib.load(base_path / "consenso_canais_scaler.pkl"),
        "consenso_canais_model": joblib.load(base_path / "consenso_canais_model.pkl"),
        "consenso_produtos_scaler": joblib.load(base_path / "consenso_produtos_scaler.pkl"),
        "consenso_produtos_model": joblib.load(base_path / "consenso_produtos_model.pkl"),
    }


def select_features_for_artifact(artifact, df: pd.DataFrame) -> pd.DataFrame:
    # Usa as features salvas no objeto (quando disponíveis) para evitar mismatch de colunas.
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

models = load_models()

# 3. Interface de Entrada de Dados (Dividida em colunas)
st.header("🛒 Dados do Cliente")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Geral & Financeiro")
    renda = st.number_input("Renda Anual", min_value=0.0, value=50000.0)
    gasto_total = st.number_input("Gasto Total", min_value=0.0, value=1500.0)
    gasto_medio = st.number_input("Gasto Médio por Compra", min_value=0.0, value=150.0)
    total_compras = st.number_input("Total de Compras", min_value=0, value=10)

with col2:
    st.subheader("Canais de Compra")
    comp_web = st.number_input("Compras na Web", min_value=0, value=3)
    comp_loja = st.number_input("Compras na Loja", min_value=0, value=5)
    comp_cat = st.number_input("Compras por Catálogo", min_value=0, value=1)
    comp_desc = st.number_input("Compras com Desconto", min_value=0, value=2)

with col3:
    st.subheader("Gastos por Produto")
    g_vinhos = st.number_input("Gasto c/ Vinhos", min_value=0.0, value=500.0)
    g_carnes = st.number_input("Gasto c/ Carnes", min_value=0.0, value=300.0)
    g_frutas = st.number_input("Gasto c/ Frutas", min_value=0.0, value=50.0)
    g_peixes = st.number_input("Gasto c/ Peixes", min_value=0.0, value=80.0)
    g_doces = st.number_input("Gasto c/ Doces", min_value=0.0, value=40.0)
    g_ouro = st.number_input("Gasto c/ Ouro (Joias)", min_value=0.0, value=100.0)

# 4. Botão de Predição
st.markdown("---")
if st.button("Classificar Cliente 🚀", type="primary", use_container_width=True):
    
    # Prepara os dados num DataFrame igual ao que o modelo espera
    dados_cliente = pd.DataFrame([{
        "Renda": renda, "Gasto_Total": gasto_total, "Gasto_Medio_Compra": gasto_medio, 
        "Total_Compras": total_compras, "Compras_Web": comp_web, "Compras_Loja": comp_loja, 
        "Compras_Catalogo": comp_cat, "Compras_Com_Desconto": comp_desc,
        "Gasto_Vinhos": g_vinhos, "Gasto_Frutas": g_frutas, "Gasto_Carnes": g_carnes, 
        "Gasto_Peixes": g_peixes, "Gasto_Doces": g_doces, "Gasto_Ouro": g_ouro
    }])
    
    try:
        cluster_geral = predict_cluster(
            dados_cliente,
            models["kmeans_geral_scaler"],
            models["kmeans_geral_model"],
        )
        cluster_canais = predict_cluster(
            dados_cliente,
            models["consenso_canais_scaler"],
            models["consenso_canais_model"],
        )
        cluster_produtos = predict_cluster(
            dados_cliente,
            models["consenso_produtos_scaler"],
            models["consenso_produtos_model"],
        )
    except Exception as exc:
        st.error("Falha ao executar a inferência. Verifique compatibilidade das features e arquivos .pkl.")
        st.exception(exc)
        st.stop()
    
    # 5. Exibição dos Resultados
    st.header("📊 Resultados da Segmentação")
    
    res_col1, res_col2, res_col3 = st.columns(3)
    
    with res_col1:
        st.success(f"**Geral (K-Means):** Cluster {cluster_geral}")
        perfil_geral = {
            0: "Baixo ticket / Alta sensibilidade a preço",
            1: "Standard / Ticket médio",
            2: "Alto valor / Maior potencial de retenção",
        }.get(cluster_geral, "Perfil não mapeado")
        st.write(f"*Perfil:* {perfil_geral}")
        
    with res_col2:
        st.info(f"**Canais (Consenso):** Cluster {cluster_canais}")
        perfil_canais = {
            0: "Comprador digital e caçador de ofertas",
            1: "Omnichannel equilibrado",
            2: "Preferência por loja física",
        }.get(cluster_canais, "Perfil não mapeado")
        st.write(f"*Perfil:* {perfil_canais}")
        
    with res_col3:
        st.warning(f"**Produtos (Consenso):** Cluster {cluster_produtos}")
        perfil_produtos = {
            0: "Foco em itens essenciais",
            1: "Focado em carnes e vinhos",
            2: "Mix premium e variedade",
        }.get(cluster_produtos, "Perfil não mapeado")
        st.write(f"*Perfil:* {perfil_produtos}")

    # Ação Direcionada (Motor de Regras)
    st.markdown("### 🎯 Recomendação de Marketing (Matriz de Ações)")
    
    # A matriz de ações é baseada na combinação dos clusters para direcionar estratégias específicas:
    if cluster_geral == 2 or cluster_produtos == 2:
        st.success("**Ação VIP:** Cliente de alto valor. Acionar contato personalizado com curadoria de luxo e eventos exclusivos.")
    elif cluster_geral == 1 and cluster_produtos == 1:
        st.info("**Ação Cross-Sell:** Enviar cupons digitais via App/E-mail com ofertas cruzadas de Vinhos e Carnes para tentar aumentar o ticket médio e fidelizar.")
    elif cluster_geral == 0 and cluster_canais == 0:
        st.warning("**Ação Retenção/Caçador:** Ativar via campanhas massivas focadas 100% em descontos nos canais digitais (E-mail/Web).")
    else:
        st.info("**Ação Padrão:** Nutrição através de newsletter com destaques semanais nas lojas físicas.")