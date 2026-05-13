import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Importaciones de tu lógica de ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# app = Flask(
#    __name__,
#    root_path=BASE_DIR,
#    ...
# )


COLUMNAS = [
'symboling','normalized_losses','make','fuel_type','aspiration',
'num_of_doors','body_style','drive_wheels','engine_location',
'wheel_base','length','width','height','curb_weight','engine_type',
'num_of_cylinders','engine_size','fuel_system','bore','stroke',
'compression_ratio','horsepower','peak_rpm','city_mpg',
'highway_mpg','price'
]

COLUMNAS_ENTRADA = [c for c in COLUMNAS if c != 'price']

_dataset_path = os.path.join(BASE_DIR, 'imports-85.data')
df = pd.read_csv(_dataset_path, header=None, names=COLUMNAS)
df = df.replace('?', np.nan)

for c in df.columns:
    try:
        df[c] = pd.to_numeric(df[c])
    except:
        pass

mediana = df['price'].median()
df['target'] = (df['price'] >= mediana).astype(int)

X = df.drop(['price','target'], axis=1)
X = pd.get_dummies(X, drop_first=True)
y = df['target']

X = X.fillna(X.median(numeric_only=True))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

modelo_lr = LogisticRegression(max_iter=5000)
modelo_lr.fit(X_train, y_train)

modelo_nn = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=5000)
modelo_nn.fit(X_train, y_train)

# Formulario mínimo: 3 categóricas + 3 numéricas (el resto de columnas se rellenan con 0 al alinear con X).
FORM_CATEGORICAS = ['make', 'body_style', 'fuel_type']
FORM_NUMERICAS = ['engine_size', 'curb_weight', 'horsepower']

BLOQUES_FORMULARIO = [
    ('Tres datos categóricos', FORM_CATEGORICAS),
    ('Tres datos numéricos', FORM_NUMERICAS),
]

ETIQUETAS = {
    'symboling': 'Índice symboling (riesgo, típico entre -3 y 3)',
    'normalized_losses': 'Pérdidas normalizadas (dejar vacío si no aplica)',
    'make': 'Fabricante (marca)',
    'fuel_type': 'Tipo de combustible',
    'aspiration': 'Aspiración del motor (std / turbo)',
    'num_of_doors': 'Número de puertas',
    'body_style': 'Tipo de carrocería',
    'drive_wheels': 'Tracción',
    'engine_location': 'Ubicación del motor',
    'wheel_base': 'Distancia entre ejes',
    'length': 'Longitud total',
    'width': 'Anchura',
    'height': 'Altura',
    'curb_weight': 'Peso en vacío (curb weight)',
    'engine_type': 'Tipo de motor',
    'num_of_cylinders': 'Número de cilindros',
    'engine_size': 'Cilindrada (engine size)',
    'fuel_system': 'Sistema de alimentación',
    'bore': 'Diámetro del cilindro (bore)',
    'stroke': 'Carrera (stroke)',
    'compression_ratio': 'Relación de compresión',
    'horsepower': 'Potencia (CV)',
    'peak_rpm': 'Revoluciones máximas (rpm)',
    'city_mpg': 'Consumo ciudad (mpg)',
    'highway_mpg': 'Consumo carretera (mpg)',
}

def _opciones_categoricas_formulario():
    out = {}
    for col in FORM_CATEGORICAS:
        serie = df[col].dropna().astype(str).str.strip()
        out[col] = sorted(serie.unique().tolist())
    return out

_OPCIONES_CAT = _opciones_categoricas_formulario()


def _norm_header(name):
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return ''
    s = str(name).strip().lower()
    s = s.replace('\ufeff', '')
    s = re.sub(r'\s+', '_', s)
    s = s.replace('-', '_')
    s = unicodedata.normalize('NFD', s)
    s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')
    s = re.sub(r'_+', '_', s).strip('_')
    return s


# Sinónimos / español / variantes → nombre canónico del dataset (Imports-85).
_ALIASES_COLUMNAS = {
    'marca': 'make', 'fabricante': 'make', 'manufacturer': 'make', 'maker': 'make',
    'combustible': 'fuel_type', 'tipo_combustible': 'fuel_type', 'fuel': 'fuel_type',
    'carroceria': 'body_style', 'tipo_carroceria': 'body_style', 'estilo': 'body_style',
    'aspiracion': 'aspiration', 'turbo': 'aspiration',
    'puertas': 'num_of_doors', 'num_puertas': 'num_of_doors', 'doors': 'num_of_doors',
    'traccion': 'drive_wheels', 'drive': 'drive_wheels',
    'ubicacion_motor': 'engine_location', 'motor_delantero_trasero': 'engine_location',
    'distancia_ejes': 'wheel_base', 'batalla': 'wheel_base',
    'largo': 'length', 'longitud': 'length',
    'ancho': 'width', 'anchura': 'width',
    'alto': 'height', 'altura': 'height',
    'peso': 'curb_weight', 'peso_vacio': 'curb_weight', 'peso_en_vacio': 'curb_weight',
    'cilindrada': 'engine_size', 'tamano_motor': 'engine_size', 'tamanio_motor': 'engine_size',
    'cc': 'engine_size',
    'potencia': 'horsepower', 'cv': 'horsepower', 'caballos': 'horsepower', 'hp': 'horsepower',
    'rpm_max': 'peak_rpm', 'rpm': 'peak_rpm',
    'consumo_ciudad': 'city_mpg', 'mpg_ciudad': 'city_mpg',
    'consumo_carretera': 'highway_mpg', 'mpg_carretera': 'highway_mpg',
    'relacion_compresion': 'compression_ratio', 'compresion': 'compression_ratio',
    'perdidas_normalizadas': 'normalized_losses', 'perdidas': 'normalized_losses',
    'riesgo': 'symboling', 'symbol': 'symboling',
    'cilindros': 'num_of_cylinders', 'n_cilindros': 'num_of_cylinders',
    'tipo_motor': 'engine_type', 'motor': 'engine_type',
    'sistema_combustible': 'fuel_system', 'alimentacion': 'fuel_system',
    'diametro_cilindro': 'bore', 'carrera': 'stroke',
    'precio': 'price', 'price_usd': 'price',
}


def _read_csv_flexible(path):
    """Intenta codificaciones y separadores habituales (Excel, Windows, UTF-8)."""
    encodings = ('utf-8-sig', 'utf-8', 'latin1', 'cp1252')
    seps = (';', ',', '\t', None)
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                kw = dict(filepath_or_buffer=path, encoding=enc, low_memory=False)
                if sep is None:
                    kw['sep'] = None
                    kw['engine'] = 'python'
                else:
                    kw['sep'] = sep
                return pd.read_csv(**kw)
            except Exception as exc:
                last_err = exc
                continue
    raise last_err if last_err else OSError('No se pudo leer el CSV')


def _mapear_columnas_a_canonicas(frame):
    """
    Construye un DataFrame con columnas canónicas (COLUMNAS_ENTRADA).
    Ignora precio y objetivo para no filtrar la predicción. Une duplicados con combine_first.
    """
    piezas = {}
    for col in frame.columns:
        key = _norm_header(col)
        if not key or key.startswith('unnamed'):
            continue
        canon = _ALIASES_COLUMNAS.get(key, key)
        if canon in ('price', 'target'):
            continue
        if canon not in COLUMNAS_ENTRADA and key in COLUMNAS_ENTRADA:
            canon = key
        if canon not in COLUMNAS_ENTRADA:
            continue
        serie = frame[col]
        if canon in piezas:
            piezas[canon] = piezas[canon].combine_first(serie)
        else:
            piezas[canon] = serie
    if not piezas:
        return pd.DataFrame(index=frame.index)
    return pd.DataFrame(piezas, index=frame.index)


def _features_desde_tabla(frame):
    """
    Completa todas las columnas de entrada; las no presentes quedan como NaN.
    Devuelve (DataFrame o None, mensaje_error o None, columnas_reconocidas_con_datos).
    """
    mapeado = _mapear_columnas_a_canonicas(frame)
    if mapeado.shape[1] == 0:
        return None, 'No se reconoció ninguna columna compatible con el modelo.', []

    usadas = [c for c in mapeado.columns if c in COLUMNAS_ENTRADA]
    if not usadas:
        return None, 'Ninguna columna coincide con el esquema esperado tras normalizar nombres.', []

    parcial = mapeado[usadas].copy()
    parcial = parcial.replace(r'^\s*$', np.nan, regex=True).replace('?', np.nan)
    con_datos = sorted([c for c in usadas if parcial[c].notna().any()])

    out = parcial.copy()
    for col in COLUMNAS_ENTRADA:
        if col not in out.columns:
            out[col] = np.nan

    for c in out.columns:
        try:
            out[c] = pd.to_numeric(out[c])
        except (ValueError, TypeError):
            pass

    return out[COLUMNAS_ENTRADA], None, con_datos


def _predecir_desde_features_raw(features_df):
    """Igual que el entrenamiento: get_dummies(drop_first=True), alinear X, escalar, LR."""
    mat = pd.get_dummies(features_df, drop_first=True)
    for col in X.columns:
        if col not in mat.columns:
            mat[col] = 0
    mat = mat[X.columns].fillna(0)
    mat = scaler.transform(mat)
    return modelo_lr.predict(mat)


def guardar_matriz(modelo, nombre, titulo_grafico):
    """Matriz de confusión sobre el conjunto de prueba (no usa los datos del formulario)."""
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(cm, cmap='Blues', vmin=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Cantidad de vehículos', rotation=270, labelpad=16, fontsize=9)

    nrows, ncols = cm.shape
    tick_labels_y = ['Precio bajo (0)', 'Precio alto (1)'][:nrows]
    tick_labels_x = ['Predicho bajo', 'Predicho alto'][:ncols]
    ax.set_xticks(np.arange(ncols))
    ax.set_yticks(np.arange(nrows))
    ax.set_xticklabels(tick_labels_x, fontsize=9)
    ax.set_yticklabels(tick_labels_y, fontsize=9)
    ax.set_xlabel('Lo que predijo el modelo', fontsize=10)
    ax.set_ylabel('Etiqueta real en prueba', fontsize=10)
    ax.set_title(titulo_grafico, fontsize=11, pad=12)

    vmax = cm.max() if cm.size else 1
    thresh = vmax / 2.0 if vmax else 0.5
    for i in range(nrows):
        for j in range(ncols):
            val = int(cm[i, j])
            ax.text(
                j, i, str(val),
                ha='center', va='center',
                color='white' if cm[i, j] > thresh else '#0c1222',
                fontsize=12, fontweight='bold',
            )

    plt.tight_layout()
    os.makedirs(os.path.join(BASE_DIR, 'static'), exist_ok=True)
    ruta_fs = os.path.join(BASE_DIR, 'static', f'{nombre}.png')
    fig.savefig(ruta_fs, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    ruta_url = f'/static/{nombre}.png'
    reporte = classification_report(
        y_test,
        y_pred,
        target_names=['Precio bajo (bajo la mediana)', 'Precio alto (sobre la mediana)'],
        zero_division=0,
    )
    return accuracy_score(y_test, y_pred), reporte, ruta_url


def _mediana_dataset(col):
    m = df[col].median()
    return float(m) if pd.notna(m) else 0.0


def _grafico_resumen_prediccion_usuario(datos, modelo_key, proba):
    """
    Gráfica generada con los valores del formulario y las probabilidades del modelo elegido.
    """
    titulo_modelo = (
        'Regresión logística' if modelo_key == 'logistica' else 'Red neuronal (MLP)'
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    fig.patch.set_facecolor('white')
    fig.suptitle(
        'Resumen gráfico según los datos que enviaste',
        fontsize=12,
        fontweight='bold',
        color='#0c1222',
        y=0.98,
    )
    cat_txt = (
        f"Marca: {datos.get('make', '—')}  ·  Carrocería: {datos.get('body_style', '—')}  ·  "
        f"Combustible: {datos.get('fuel_type', '—')}"
    )
    fig.text(0.5, 0.90, cat_txt, ha='center', fontsize=10, color='#333333')

    labels_num = ['Cilindrada', 'Peso (lb)', 'Potencia (CV)']
    cols_num = ['engine_size', 'curb_weight', 'horsepower']
    vals = []
    meds = []
    for col in cols_num:
        try:
            vals.append(float(datos.get(col)))
        except (TypeError, ValueError):
            vals.append(0.0)
        meds.append(_mediana_dataset(col))

    xpos = np.arange(len(labels_num))
    w = 0.36
    axes[0].bar(xpos - w / 2, vals, w, label='Tus datos', color='#2ab8a8', edgecolor='#0c1222', linewidth=0.4)
    axes[0].bar(xpos + w / 2, meds, w, label='Mediana del dataset', color='#9aa3b5', edgecolor='#0c1222', linewidth=0.4)
    axes[0].set_xticks(xpos)
    axes[0].set_xticklabels(labels_num, fontsize=9)
    axes[0].set_ylabel('Valor')
    axes[0].legend(fontsize=8, loc='upper right')
    axes[0].set_title('Tres números del formulario', fontsize=10, color='#0c1222')
    axes[0].grid(axis='y', linestyle='--', alpha=0.35)

    p_bajo, p_alto = float(proba[0]), float(proba[1])
    axes[1].bar(
        ['Prob. precio bajo', 'Prob. precio alto'],
        [p_bajo, p_alto],
        color=['#f0b429', '#2ab8a8'],
        edgecolor='#0c1222',
        linewidth=0.4,
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel('Probabilidad')
    axes[1].set_title(f'Confianza — {titulo_modelo}', fontsize=10, color='#0c1222')
    for i, v in enumerate([p_bajo, p_alto]):
        axes[1].text(i, v + 0.03, f'{v:.0%}', ha='center', fontsize=10, fontweight='bold', color='#0c1222')
    axes[1].grid(axis='y', linestyle='--', alpha=0.35)

    plt.tight_layout(rect=[0, 0, 1, 0.86])
    os.makedirs(os.path.join(BASE_DIR, 'static'), exist_ok=True)
    ruta_fs = os.path.join(BASE_DIR, 'static', 'ultima_prediccion.png')
    fig.savefig(ruta_fs, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)

@app.route('/')
def index():
    return render_template(
        'index.html',
        bloques=BLOQUES_FORMULARIO,
        etiquetas=ETIQUETAS,
        opciones=_OPCIONES_CAT,
        categoricas=set(FORM_CATEGORICAS),
    )

@app.route('/metricas')
def metricas():
    acc1, rep1, img1 = guardar_matriz(
        modelo_lr, 'logistica', 'Matriz de confusión — Regresión logística',
    )
    acc2, rep2, img2 = guardar_matriz(
        modelo_nn, 'neuronal', 'Matriz de confusión — Red neuronal',
    )

    return render_template(
        'metricas.html',
        acc1=acc1,
        rep1=rep1,
        img1=img1,
        acc2=acc2,
        rep2=rep2,
        img2=img2,
        cache_bust=int(time.time() * 1000),
    )

@app.route('/predecir', methods=['POST'])
def predecir():
    datos = request.form.to_dict()
    modelo = datos.pop('modelo', 'logistica')

    nuevo = pd.DataFrame([datos])
    nuevo = nuevo.replace(r'^\s*$', np.nan, regex=True)
    nuevo = nuevo.replace('?', np.nan)

    for c in nuevo.columns:
        try:
            nuevo[c] = pd.to_numeric(nuevo[c])
        except (ValueError, TypeError):
            pass

    nuevo = pd.get_dummies(nuevo, drop_first=True)

    for col in X.columns:
        if col not in nuevo.columns:
            nuevo[col] = 0

    nuevo = nuevo[X.columns]
    nuevo = nuevo.fillna(0)

    nuevo = scaler.transform(nuevo)

    if modelo == 'neuronal':
        clf = modelo_nn
        pred = int(clf.predict(nuevo)[0])
    else:
        clf = modelo_lr
        pred = int(clf.predict(nuevo)[0])

    proba = clf.predict_proba(nuevo)[0]
    _grafico_resumen_prediccion_usuario(datos, modelo, proba)

    es_alto = pred == 1
    img_resumen = url_for('static', filename='ultima_prediccion.png') + '?t=' + str(int(time.time() * 1000))
    return render_template('resultado.html', es_alto=es_alto, img_resumen=img_resumen)

@app.route('/lotes', methods=['GET','POST'])
def lotes():
    resultados = None
    error = None
    info = None

    if request.method == 'POST':
        archivo = request.files.get('archivo')
        if not archivo or not archivo.filename:
            error = 'No se seleccionó ningún archivo.'
        else:
            uploads_dir = os.path.join(BASE_DIR, 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            nombre_seguro = secure_filename(archivo.filename) or 'datos.csv'
            ruta = os.path.join(uploads_dir, nombre_seguro)
            archivo.save(ruta)
            try:
                crudo = _read_csv_flexible(ruta)
                if crudo.shape[0] == 0:
                    error = 'El archivo no contiene filas de datos.'
                else:
                    features, err, reconocidas = _features_desde_tabla(crudo)
                    if err:
                        error = err
                    else:
                        pred = _predecir_desde_features_raw(features)
                        resultados = pred.tolist()
                        info = (
                            f'Se procesaron {len(crudo)} filas. '
                            f'Columnas reconocidas con al menos un valor: '
                            f'{", ".join(reconocidas) or "ninguna (todo ausente: predicción muy orientativa)"}. '
                            'El resto de variables se consideran no informadas (ceros tras codificación).'
                        )
            except Exception as exc:
                error = f'No se pudo interpretar el archivo: {exc}'
            finally:
                try:
                    os.remove(ruta)
                except OSError:
                    pass

    return render_template(
        'lotes.html',
        resultados=resultados,
        error=error,
        info=info,
    )
# --- CODIGO CORREGIDO PARA STREAMLIT ---

# Comentamos las funciones de Flask que no funcionan aquí
# @app.route('/compartir')
# def compartir():
#    return render_template('compartir.html')

# En lugar de app.run(), usamos comandos de Streamlit para mostrar resultados
if resultados:
    st.success("¡Predicción completada!")
    st.write("### Resultados:")
    st.write(resultados)
    
    if info:
        st.info(info)

if error:
    st.error(error)

# El bloque if __name__ == '__main__': ya no necesita app.run()
if __name__ == '__main__':
    st.write("La aplicación está lista y corriendo en Streamlit.")
