import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Importaciones de lógica de ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Predicción de Precios de Autos", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- CONSTANTES Y ESQUEMA ---
COLUMNAS = [
    'symboling','normalized_losses','make','fuel_type','aspiration',
    'num_of_doors','body_style','drive_wheels','engine_location',
    'wheel_base','length','width','height','curb_weight','engine_type',
    'num_of_cylinders','engine_size','fuel_system','bore','stroke',
    'compression_ratio','horsepower','peak_rpm','city_mpg',
    'highway_mpg','price'
]

ETIQUETAS = {
    'symboling': 'Índice symboling (riesgo, -3 a 3)',
    'make': 'Fabricante (marca)',
    'fuel_type': 'Tipo de combustible',
    'body_style': 'Tipo de carrocería',
    'curb_weight': 'Peso en vacío (lb)',
    'engine_size': 'Cilindrada (engine size)',
    'horsepower': 'Potencia (CV)',
}

# --- CARGA Y ENTRENAMIENTO ---
@st.cache_resource
def inicializar_modelo():
    _dataset_path = os.path.join(BASE_DIR, 'imports-85.data')
    if not os.path.exists(_dataset_path):
        return None, None, None, None, None

    df_raw = pd.read_csv(_dataset_path, header=None, names=COLUMNAS)
    df_raw = df_raw.replace('?', np.nan)

    for c in df_raw.columns:
        try: df_raw[c] = pd.to_numeric(df_raw[c])
        except: pass

    # Definir objetivo: 1 si es caro (>= mediana), 0 si es barato
    mediana_precio = df_raw['price'].median()
    df_raw['target'] = (df_raw['price'] >= mediana_precio).astype(int)

    # Preprocesamiento
    X_data = df_raw.drop(['price','target'], axis=1)
    X_data = pd.get_dummies(X_data, drop_first=True)
    X_data = X_data.fillna(X_data.median(numeric_only=True))
    y_data = df_raw['target']

    # Guardar nombres de columnas originales del entrenamiento
    columnas_entrenamiento = X_data.columns

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    
    lr = LogisticRegression(max_iter=5000).fit(X_train_sc, y_train)
    nn = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=5000).fit(X_train_sc, y_train)
    
    return lr, nn, sc, columnas_entrenamiento, df_raw

modelo_lr, modelo_nn, scaler, X_columns, df = inicializar_modelo()

# --- FUNCIONES DE APOYO ---
def predecir_interfaz(datos_dict, modelo_elegido):
    nuevo = pd.DataFrame([datos_dict])
    nuevo = pd.get_dummies(nuevo)
    # Alineación estricta con las columnas de entrenamiento
    nuevo = nuevo.reindex(columns=X_columns, fill_value=0)
    nuevo_sc = scaler.transform(nuevo)
    
    clf = modelo_nn if modelo_elegido == 'Red Neuronal' else modelo_lr
    pred = clf.predict(nuevo_sc)[0]
    proba = clf.predict_proba(nuevo_sc)[0]
    return pred, proba

# --- INTERFAZ STREAMLIT ---
st.title("🚗 Predicciones del Sector Automotriz")

if df is None:
    st.error("No se encontró el archivo 'imports-85.data'. Por favor súbelo a la raíz de tu proyecto.")
else:
    tabs = st.tabs(["Individual", "Carga por Lotes", "Métricas del Modelo"])

    # --- TAB 1: PREDICCIÓN INDIVIDUAL ---
    with tabs[0]:
        st.header("Simulador de Precio")
        with st.form("form_pred"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Datos Numéricos")
                e_size = st.number_input(ETIQUETAS['engine_size'], value=int(df['engine_size'].median()))
                c_weight = st.number_input(ETIQUETAS['curb_weight'], value=int(df['curb_weight'].median()))
                h_power = st.number_input(ETIQUETAS['horsepower'], value=int(df['horsepower'].median()))
            with col2:
                st.subheader("Datos Categóricos")
                make = st.selectbox(ETIQUETAS['make'], sorted(df['make'].unique().tolist()))
                body = st.selectbox(ETIQUETAS['body_style'], sorted(df['body_style'].unique().tolist()))
                fuel = st.selectbox(ETIQUETAS['fuel_type'], sorted(df['fuel_type'].unique().tolist()))
            
            modelo_sel = st.radio("Modelo a utilizar:", ["Regresión Logística", "Red Neuronal"])
            btn_pred = st.form_submit_button("Realizar Predicción")

        if btn_pred:
            datos_usuario = {
                'engine_size': e_size, 'curb_weight': c_weight, 'horsepower': h_power,
                'make': make, 'body_style': body, 'fuel_type': fuel
            }
            pred, proba = predecir_interfaz(datos_usuario, modelo_sel)
            st.divider()
            if pred == 1:
                st.warning("### Resultado: Precio ALTO (Sobre la mediana)")
            else:
                st.success("### Resultado: Precio BAJO (Bajo la mediana)")
            
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.barh(['Prob. Bajo', 'Prob. Alto'], proba, color=['#f0b429', '#2ab8a8'])
            ax.set_xlim(0, 1)
            st.pyplot(fig)

    # --- TAB 2: CARGA POR LOTES ---
    with tabs[1]:
        st.header("Predicción Masiva (.CSV)")
        archivo_subido = st.file_uploader("Sube tu archivo CSV", type=["csv"])
        if archivo_subido:
            df_lote = pd.read_csv(archivo_subido)
            if st.button("Procesar Lote"):
                df_proc = pd.get_dummies(df_lote).reindex(columns=X_columns, fill_value=0)
                df_proc_sc = scaler.transform(df_proc)
                predicciones = modelo_lr.predict(df_proc_sc)
                df_lote['Resultado'] = ["Caro" if p == 1 else "Barato" for p in predicciones]
                st.dataframe(df_lote)

    # --- TAB 3: MÉTRICAS (CORREGIDO) ---
    with tabs[2]:
        st.header("Rendimiento de los Modelos")
        
        # Preparación global de los datos para evaluación
        X_eval = df.drop(['price', 'target'], axis=1)
        X_eval = pd.get_dummies(X_eval).reindex(columns=X_columns, fill_value=0)
        X_eval_sc = scaler.transform(X_eval)
        y_real = df['target']

        m_col1, m_col2 = st.columns(2)
        modelos_lista = [(modelo_lr, "Regresión Logística"), (modelo_nn, "Red Neuronal")]

        for i, (m, nombre) in enumerate(modelos_lista):
            with (m_col1 if i == 0 else m_col2):
                st.subheader(f"Modelo: {nombre}")
                
                preds = m.predict(X_eval_sc)
                acc = accuracy_score(y_real, preds)
                
                st.metric("Exactitud (Accuracy)", f"{acc:.2%}")
                st.text("Reporte de Clasificación:")
                st.code(classification_report(y_real, preds))
                
                # Matriz de Confusión Visual
                cm = confusion_matrix(y_real, preds)
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.imshow(cm, cmap='Blues', interpolation='nearest')
                ax.set_title(f"Matriz de Confusión - {nombre}")
                st.pyplot(fig)

st.write("---")
st.caption("Aplicación corregida y migrada a Streamlit.")
