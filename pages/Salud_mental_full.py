# salud .py
'''
Analisis de  salud mental  basado en licencias médicas otorgadas por la Superintendencia de Seguridad Social de Chile.
'''
# librerias
import os
import pandas as pd
import numpy as np
import time
import streamlit as st
import streamlit.components.v1 as components
from streamlit_extras.metric_cards import style_metric_cards

import base64
from io import BytesIO
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import StrMethodFormatter
from millify import millify
from pygwalker.api.streamlit import StreamlitRenderer

from datetime import datetime
from funciones import menu_pages, convert_df
import duckdb
import hashlib

# configuration
st.set_page_config(
    page_title="Salud mental",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def hash_value(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


# ESTILOS
with open('style/style.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

no_sidebar_style = """
    <style>
        div[data-testid="stSidebarNav"] {display: none;}
    </style>
"""
st.markdown(no_sidebar_style, unsafe_allow_html=True)

menu_pages()

# public = "https://vgclabfiles.blob.core.windows.net/public/"
public = "data/"
filename = public+"licencias.csv"

columnas = ['FECHA', 'LICENCIA', 'FORMULARIO', 'TRABAJADOR', 'EMISION', 'INICIO', 'DIAS', 'CODIGO', 'TIPO', 'PROFESIONAL', 'ENTIDAD', 'CAJA', 'ESTADO', 'DESDE', 'HASTA',
            'DIASAUT', 'SUBSIDIO', 'CODAUT', 'AUTORIZACION', 'CODPREV', 'RECHAZO', 'SUBCIE10', 'GRUPOCIE10', 'CIE10', 'LIQUIDO', 'SUSESO']
dtipos = {
    'FECHA': np.int16,
    'LICENCIA': str,
    'TRABAJADOR': str,
    'PROFESIONAL': str,
    'DIAS': np.int16,
    'DIASAUT': np.int16,
    'CODIGO': str,
    'CODAUT': str,
    'CODPREV': str,
}

month_names = {
    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
    5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
    9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
}
month_shorts = {
    1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr',
    5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Ago',
    9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
}

progress_text = "Cargando datos de salud mental..."
pro_bar = st.progress(0.15, text=progress_text)

df = pd.read_csv(filename, encoding='utf-8', sep=",",
                 na_values='NA',
                 usecols=columnas,
                 dtype=dtipos
                 )


df['LIQUIDO'] = df['LIQUIDO'].fillna(0).astype(int)
df['SUSESO'] = df['SUSESO'].fillna(0).astype(int)
df['EMISION'] = pd.to_datetime(df['EMISION']).dt.date
df['DESDE'] = pd.to_datetime(df['DESDE']).dt.date
df['HASTA'] = pd.to_datetime(df['HASTA']).dt.date
df['FECHA'] = pd.to_datetime(df['EMISION']).dt.date
st.sidebar.subheader("Licencias médicas ")
# Filtro de rango de fechas
min_date = df['EMISION'].min()
max_date = df['EMISION'].max()
date_start = st.sidebar.date_input(
    'Emisión desde', value=pd.to_datetime(min_date))
date_end = st.sidebar.date_input(
    'Emisión hasta', value=pd.to_datetime(max_date))

df = df[
    (df['EMISION'] >= pd.to_datetime(date_start)) &
    (df['EMISION'] <= pd.to_datetime(date_end))]


# df['EMISION'] = df['EMISION'].dt.date

# Filtro por SUBCIE10
# cie10 = df['CIE10'].unique().tolist()
# cie10.insert(0, "Todos")
# cie10_to_filter = st.sidebar.multiselect(
#     'CIE10', cie10, default="Todos")
# if cie10_to_filter is not 'Todos':
#     df = df[df["CIE10"] == cie10_to_filter]
# else:
#     pass

pro_bar.progress(0.3, text="Estableciendo métricas...")
# --------------------------
# METRICAS
# --------------------------

total_records = df.shape[0]
total_licencias = df["LICENCIA"].nunique()
total_trabajadores = df["TRABAJADOR"].nunique()
total_profesional = df["PROFESIONAL"].nunique()
total_rechazadas = df[df["ESTADO"] == "RECHAZADA"].shape[0]
total_autorizadas = df[df["ESTADO"] == "CONFIRMADA"].shape[0]

# --------------------------
# MAIN
# --------------------------
st.subheader(f"Licencias por salud mental")

tabPanel, tabTable, tabBleau,  tabInfo = st.tabs(
    ["Panel", " Tabla", ":bar_chart: Informes", "Información"])

with tabPanel:
    col1, col2, col3, col4, col5 = st.columns(5, gap="medium")
    col1.metric("Licencias", millify(total_licencias, precision=2))
    col2.metric("Rechazadas", millify(total_rechazadas, precision=2))
    col3.metric("Autorizadas", millify(total_autorizadas, precision=2))
    col4.metric("Solicitantes", millify(total_trabajadores, precision=2))
    col5.metric("Profesionales", millify(total_profesional, precision=2))

    style_metric_cards()

    pro_bar.progress(0.5, text="Procesando gráficos...")
    # ------------------------------------------------------------
    # ACTORES DEMANDANTES Y AFECTADOS
    # ------------------------------------------------------------
    st.divider()
    # -------------------------------------------
    st.write(df.head())
    # -------------------------------------------
    st.divider()
    coll, colr = st.columns(2, gap="medium")

    with coll:
        fig = px.pie(df, names='ESTADO',
                     title='Estado de las licencias')
        fig.update_layout(
            title={
                'xanchor': 'center', 'yanchor': 'top', 'y': .95, 'x': 0.5, 'font': {'size': 18}})
        st.plotly_chart(fig, use_container_width=True)

    with colr:
        fig = px.pie(df, names='TIPO', title='Tipo de licencia')
        fig.update_layout(
            title={
                'xanchor': 'center', 'yanchor': 'top', 'y': .95, 'x': 0.5, 'font': {'size': 18}})
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------
    # TENDENCIAS DE PAGOS DE LICENCIAS
    # ------------------------------------------------------------

    # Create a line plot for 'LIQUIDO' and 'SUSESO' based on 'FECHA de emision'
    df['EMISION'] = pd.to_datetime(df['EMISION'])
    df['YEAR_MONTH'] = df['EMISION'].dt.strftime('%Y-%m')

    # Aggregate data by year and month
    monthly_data = df.groupby('YEAR_MONTH').agg(
        {'LIQUIDO': 'sum', 'SUSESO': 'sum'}).reset_index()
    monthly_data['YEAR_MONTH'] = monthly_data['YEAR_MONTH'].astype(str)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=monthly_data['YEAR_MONTH'], y=monthly_data['LIQUIDO'],
                             mode='lines+markers',
                             name='LIQUIDO'))

    fig.add_trace(go.Scatter(x=monthly_data['YEAR_MONTH'], y=monthly_data['SUSESO'],
                             mode='lines+markers',
                             name='SUSESO'))

    fig.update_layout(title={'text': 'Tendencias de pagos de licencia', 'xanchor': 'center',
                             'yanchor': 'top', 'y': 0.95, 'x': 0.5, 'font': {'size': 18}},
                      xaxis_title='Año y Mes',
                      yaxis_title='Monto de Pago',
                      xaxis_tickangle=-45)

    st.plotly_chart(fig, use_container_width=True)
    # ----------------------------------------------

    # ------------------------------------------------------------
    # EMISION DE PAGOS DE LICENCIAS
    # ------------------------------------------------------------

    # ----------------------------------------------
    # Create a bar plot for 'LIQUIDO' and 'SUSESO' based on 'FECHA'

    # ------------------------------------------------------------
    # EMISION DE PAGOS DE LICENCIAS EN AÑOS
    # ------------------------------------------------------------
    st.divider()

    df_bar = df.groupby('FECHA')[['LIQUIDO', 'SUSESO']].sum().reset_index()
    fig_bar = px.bar(df_bar, x='FECHA', y=[
                     'LIQUIDO', 'SUSESO'], title='Tendencias de pagos en Años')
    fig_bar.update_layout(yaxis_title=None, xaxis_title='FECHA',
                          title={
                              'xanchor': 'center', 'yanchor': 'top', 'y': .95, 'x': 0.5, 'font': {'size': 18}},
                          legend=dict(title=None,
                                      yanchor='top', xanchor='center', x=0.5))
    st.plotly_chart(fig_bar, use_container_width=True)

    # ----------------------------------------------
    # Configuraciones de visualización
    st.write("Tendencias a lo largo del tiempo para la emisión de licencias médicas.")
    st.divider()
    sns.set(style="whitegrid")

    # 1. Tendencias a lo largo del tiempo para 'EMISION'
    df['Año'] = df['EMISION'].dt.year
    df['Mes'] = df['EMISION'].dt.month

    # Agrupar por año y mes para contar el número de licencias emitidas
    licencias_por_mes = df.groupby(
        ['Año', 'Mes']).size().reset_index(name='Cantidad')

    # Visualización
    plt.figure(figsize=(15, 6))
    sns.lineplot(data=licencias_por_mes, x='Año', y='Cantidad',
                 estimator='sum', ci=None, marker='o')
    plt.title('Tendencia de Licencias Médicas Emitidas por Año', fontsize=18)
    plt.xlabel('Año', fontsize=14)
    plt.ylabel('Cantidad de licencias emitidas', fontsize=14)
    plt.xticks(licencias_por_mes['Año'].unique())
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    st.pyplot(plt)

    # ----------------------------------------------
    # 2. Distribución del estado de las licencias médicas ('ESTADO')
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y='ESTADO',
                  order=df['ESTADO'].value_counts().index)
    plt.title('Distribución de Estados de Licencias Médicas')
    plt.xlabel('Cantidad')
    plt.ylabel('Estado')
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    st.pyplot(plt)

    # ----------------------------------------------
    # 3. Análisis de la columna 'AUTORIZACION'
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y='AUTORIZACION',
                  order=df['AUTORIZACION'].value_counts().index)
    plt.title('Distribución de Autorizaciones de Licencias Médicas')
    plt.xlabel('Cantidad')
    plt.ylabel('Autorización')
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    st.pyplot(plt)

    st.divider()
    # ----------------------------------------------
    coll, colr = st.columns(2, gap="medium")
    with coll:
        st.write("La visualización corregida muestra la tendencia mensual de licencias médicas emitidas en el período. A través de esta gráfica, podemos observar cómo la cantidad de licencias varía mes a mes a lo largo del período de estudio.")
    with colr:
        st.write("Podría haber patrones estacionales evidentes, donde ciertos meses muestran variación en el número de licencias emitidas. Estos patrones podrían estar relacionados con factores estacionales como enfermedades comunes en ciertas épocas del año.")
    st.write("")

    st.divider()
    # ----------------------------------------------
    # variables estacionales
    licencias_por_mes_promedio = df.groupby(
        'Mes').size().reset_index(name='Cantidad')
    licencias_por_mes_promedio['CantidadPromedio'] = licencias_por_mes_promedio['Cantidad'] / len(
        df['Año'].unique())

    # Visualización

    plt.figure(figsize=(12, 6))
    sns.barplot(data=licencias_por_mes_promedio, x='Mes',
                y='CantidadPromedio', palette='coolwarm')
    plt.title('Promedio de Licencias Médicas Emitidas por Mes', fontsize=16)
    plt.xlabel('Mes', fontsize=14)
    plt.ylabel('Promedio de Licencias Médicas Emitidas', fontsize=14)
    plt.xticks(range(0, 12), ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
               'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'])
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    plt.tight_layout()
    st.pyplot(plt)

    # ----------------------------------------------
    # codigos CIE10
    # Identificar los 10 códigos de diagnóstico (CIE10) más comunes
    st.subheader("CIE10")
    st.write("Patrones específicos por condición de salud: Ciertos códigos de diagnóstico muestran picos claros en meses específicos, lo que puede indicar la estacionalidad de estas condiciones. Por ejemplo, si los códigos relacionados con enfermedades respiratorias aumentan en los meses de invierno, esto sugeriría una influencia estacional.")
    st.write("")
    st.write("La categoría F de la CIE-10 incluye trastornos mentales y del comportamiento. A continuación, se muestra la tendencia mensual de los 10 códigos de diagnóstico más comunes en esta categoría.")
    st.markdown("""
    - F00: Trastornos mentales y del comportamiento debidos al uso de alcohol
    - F01: Demencia en la enfermedad de Alzheimer, tipo Alzheimer
    - F02: Demencia en otras enfermedades clasificadas en otra parte
    - F03: Demencia, no especificada
    - F04: Síndrome amnésico orgánico no inducido por alcohol
    - F05: Delirio no inducido por alcohol
    - F06: Otros trastornos mentales debidos a lesión y disfunción cerebral y a enfermedad somática
    - F07: Trastornos de la personalidad y del comportamiento en adultos
    - F10: Trastornos mentales y del comportamiento debidos al uso de alcohol
    - F14: Trastornos mentales y del comportamiento debidos al uso de cocaína
    - F19: Trastornos mentales y del comportamiento debidos al uso de múltiples drogas y al uso de otras sustancias psicoactivas
    - F20: Esquizofrenia
    - F22: Trastorno delirante persistente
    - F25: Trastornos esquizoafectivos
    - F064: Trastorno de ansiedad, orgánico
    - F065: Trastorno disociativo, orgánico
    - F070: Trastorno de personalidad, orgánico
    - F102: Trastornos mentales y del comportamiento debidos al uso de alcohol síndrome de dependencia
    - F141: Trastornos mentales y del comportamiento debidos al uso de cocaína, uso nocivo
    """)

    top_cie10 = df['CIE10'].value_counts().nlargest(10).index.tolist()

    # Filtrar los datos para incluir solo los registros con los códigos de diagnóstico más comunes
    data_top_cie10 = df[df['CIE10'].isin(top_cie10)]

    # ----------------------------------------------
    # TENDENCIA MENSUAL DE LICENCIAS POR CIE10
    # 2024-03-31
    # ----------------------------------------------
    st.divider()

    df['EMISION'] = pd.to_datetime(df['EMISION'])

    licencias_por_cie = df.groupby(
        ['YEAR_MONTH', 'CIE10']).size().reset_index(name='Cantidad')
    licencias_por_cie['YEAR_MONTH'] = licencias_por_cie['YEAR_MONTH'].astype(
        str)
    fig_cie = px.line(licencias_por_cie, x='YEAR_MONTH', y='Cantidad', color='CIE10',
                      title='Tendencia Mensual de Licencias por Tipo CIE10', markers=True,
                      # hovertemplate='Fecha: {x}<br>Licencias: {y}'
                      )
    fig_cie.update_traces(hovertemplate='Fecha: %{x}<br>Licencias: %{y}')
    fig_cie.update_xaxes(tickangle=-90, ticks="outside",
                         nticks=48, showgrid=True, showline=True)
    fig_cie.update_yaxes(showgrid=True, showline=True)
    fig_cie.update_layout(yaxis_title='Cantidad de licencias emitidas',
                          xaxis_title='Fecha',
                          height=600,
                          title={
                              'xanchor': 'center', 'yanchor': 'top', 'y': .95, 'x': 0.5, 'font': {'size': 18}})
    st.plotly_chart(fig_cie, use_container_width=True)

    # ------------------------------------------------------------
    # LICENCIAS POR SUBCIE10
    # ------------------------------------------------------------
    st.divider()

    licencias_por_cie = df.groupby(
        ['YEAR_MONTH', 'SUBCIE10']).size().reset_index(name='Cantidad')
    licencias_por_cie['YEAR_MONTH'] = licencias_por_cie['YEAR_MONTH'].astype(
        str)
    fig_cie = px.line(licencias_por_cie, x='YEAR_MONTH', y='Cantidad', color='SUBCIE10',
                      title='Tendencia Mensual de Licencias por SUBCIE10', markers=True,
                      # hovertemplate='Fecha: {x}<br>Licencias: {y}'
                      hover_name='SUBCIE10',
                      )
    fig_cie.update_traces(
        hovertemplate='Fecha: %{x}<br>Licencias: %{y}')
    fig_cie.update_xaxes(tickangle=-90, ticks="outside",
                         nticks=48, showgrid=True, showline=True)
    fig_cie.update_yaxes(showgrid=True, showline=True)
    fig_cie.update_layout(yaxis_title='Cantidad de licencias emitidas',
                          xaxis_title='Fecha',
                          title={
                              'xanchor': 'center', 'yanchor': 'top', 'y': .95, 'x': 0.5, 'font': {'size': 18}})
    st.plotly_chart(fig_cie, use_container_width=True)

    coll, colr = st.columns(2, gap="medium")
    coll.write("Diferencias en los patrones estacionales: No todas las condiciones de salud siguen el mismo patrón estacional, lo cual es esperado. Algunas condiciones pueden ser consistentemente altas a lo largo del año, mientras que otras muestran variabilidad mensual significativa.")
    colr.write("Importancia del contexto clínico y social: Para interpretar estos patrones correctamente, es crucial considerar el contexto clínico de cada código de diagnóstico (CIE10) y posibles factores sociales o ambientales que puedan influir en estas tendencias.")

    st.markdown(""" ### Ejemplo de interpretación
Los gráficos muestran una clara tendencia al alza en la cantidad de licencias médicas desde 2018 hasta 2021, con picos notables en junio y diciembre de cada año.

Análisis de tendencias: la tendencia al alza podría sugerir una prevalencia creciente de problemas de salud o un acceso más fácil a las licencias médicas.

Patrones estacionales: los picos en junio y diciembre podrían indicar enfermedades estacionales, como brotes de gripe en invierno y alergias a fines de la primavera.

Comparaciones anuales: si diciembre de 2020 muestra un pico mucho más alto en comparación con los diciembres anteriores, podría correlacionarse con un evento en particular, como la pandemia de COVID-19.

Fluctuaciones mensuales: las cifras bajas constantes en febrero podrían sugerir menos problemas de salud o menos acceso a la atención médica durante ese mes.

Volumen: los volúmenes altos en ciertos meses podrían indicar una necesidad de mayores servicios o recursos de atención médica durante esos períodos.""")


# -------------------------------------------
with tabTable:
    dftabla = df.drop(columns=["LICENCIA", "FECHA",
                      "TRABAJADOR", "PROFESIONAL", "YEAR_MONTH", "Año", "Mes"])
    dftabla['fecha'] = df['EMISION'].dt.date
    dftabla['EMISION'] = df['EMISION'].dt.date
    # dftabla['DESDE'] = df['DESDE'].dt.date
    # dftabla['HASTA'] = df['HASTA'].dt.date

    st.dataframe(dftabla, height=600)

    with st.sidebar:
        st.download_button("Descargar datos (csv)",
                           convert_df(dftabla), "licencias.csv",
                           "text/csv", key="donwload-csv")

# --------------------------
with tabBleau:  # graficos personalizados
    # @st.cache_resource
    def get_pyg_renderer() -> "StreamlitRenderer":
        return StreamlitRenderer(df, spec_io_mode="rw")

    renderer = get_pyg_renderer()
    renderer.explorer()

    # -------------------------------------------

pro_bar.empty()
# -------------------------------------------
# with tabBleau:
#     st.write("Análisis con Pywalker")
# report = pgw.walk(df, return_html=True)
# components.html(report, height=1000, scrolling=True)

# -------------------------------------------
with tabInfo:
    st.write("Información")
    st.write(
        "Analisis de  salud mental  basado en licencias médicas otorgadas por la [Superintendencia de Seguridad Social de Chile](https://www.suseso.cl/).")
    st.markdown('''El conjunto de datos contiene una variedad de columnas que ofrecen información detallada sobre las licencias médicas otorgadas por diagnóstico CIE-10 de salud mental. A continuación, se detalla una descripción inicial de las columnas relevantes para el análisis exploratorio:

- FECHA: Año de la emisión de la licencia.
- LICENCIA: Número de licencia médica.
- FORMULARIO: Tipo de formulario de licencia (electrónico, papel, etc.).
- TRABAJADOR: Identificación del trabajador.
- EMISION: Fecha de emisión de la licencia.
- INICIO: Fecha de inicio de la licencia.
- DIAS: Duración en días de la licencia.
- TIPO: Tipo de licencia (por enfermedad, accidente no laboral, etc.).
- CIE10: Código CIE-10 del diagnóstico.
- GRUPOCIE10: Grupo del diagnóstico CIE-10.
- SUBCIE10: Subcategoría del diagnóstico CIE-10.

''')
