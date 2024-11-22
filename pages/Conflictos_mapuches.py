'''
Conflictos mapuches en Chile basadod en datos de MACEDA (Mapuche Chilean State Conflict Event Database).
Fecha: 2024-01-12
Autor: Patricio Araneda
2024-10-10: 
cambio de color de gráficos a modelo de color prism
carga de datos desde contenedor https://vgclabfiles.blob.core.windows.net/public/
net stop winnat si hay errores de acceso por puertos usados.
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

from millify import millify
from pygwalker.api.streamlit import StreamlitRenderer

from datetime import datetime
from funciones import menu_pages, convert_df, logo
import duckdb

from matplotlib import pyplot as plt
import re

# configuration
st.set_page_config(
    page_title="Maceda",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


def base64_to_image(base64_string):
    # Decode the base64 string
    byte_data = base64.b64decode(base64_string)

    # Use BytesIO to convert the byte data to image
    return Image.open(BytesIO(byte_data))


menu_pages()


progress_text = "Cargando datos..."
pro_bar = st.progress(0.10, text=progress_text)

# carga de datos
public = "https://vgclabfiles.blob.core.windows.net/public/"


# df = pd.read_csv(public+"dataset_maceda.csv", sep=",", low_memory=False, index_col=0)

# Using smart_open to load the data
filename = public + "dataset_maceda.csv"

# with open(filename, 'rb') as f:
#    df = pd.read_csv(f, sep=",", low_memory=False,
#                     index_col=0, encoding='CP1252')

# pandas large files
reader = pd.read_csv(filename, chunksize=10000, nrows=10000)
df = pd.concat([x for x in reader], ignore_index=True)

# try:
#     conn = duckdb.connect('vglab.db')
#     query = "SELECT * FROM maceda"
#     df = conn.execute(query).fetchdf()
#     conn.close()
# except:
#     st.error("Error al cargar los datos")
#     st.stop()


# df['fecha'] = pd.to_datetime(df['año'].astype(
#     str) + '-' + df['mes'].astype(str) + '-' + df['dia'].astype(str), errors='coerce')
# df['fecha'] = pd.to_datetime(
#     df['fecha'], format='mixed', dayfirst=True).dt.date
# df['fecha'] = df['fecha'].dt.date
#

# sacar columnasS
df = df.drop(columns=['id_evento_relacionado', 'fecha_reportada'])

df = df.rename(columns={'ubicacion_tipo': 'ubicacion',
                        'evento_tipo_maceda': 'tipo de evento',
                        'evento_especifico': 'detalle de evento',
                        'actor_tipo_1': 'tipo de actor',
                        'actor_especifico_1': 'actor principal',
                        'actor_especifico_1_armas': 'armas',
                        'actor_tipo_2': 'actor afectado',
                        'actor_mapuche': 'actor mapuche',
                        'mapuche_identificado': 'mapuche identificado',
                        'mesn': 'mes'})

df['fecha'] = pd.to_datetime(df['fecha'], format='%Y-%m-%d')
# cambiar el orden de fecha al inicio
fecha = df.pop("fecha")
df.insert(0, "fecha", fecha)

pro_bar.progress(0.3, text="Estableciendo métricas...")

# crear dataframe estatico, independiente de selecciones del usuario
serie = df.groupby(['fecha', 'tipo de evento']
                   ).size().reset_index(name='cuenta')
# medios = df.sort_values(by="medio").medio.unique()
# medios.insert(0, "Todos")

# sidebar
years = df["fecha"].dt.year.unique().tolist()
years.insert(0, "Todos")
anual = st.sidebar.selectbox("Seleccione un año", years)


if anual != 'Todos':
    df = df[df["fecha"].dt.year == anual]
else:
    pass

tipos_eventos = df["tipo de evento"].unique().tolist()
tipo_evento = st.sidebar.multiselect(
    "Seleccione un tipo de evento", tipos_eventos, default=tipos_eventos)

if tipo_evento != None:
    df = df[df["tipo de evento"].isin(tipo_evento)]
else:
    df = df


regiones = df["region"].unique().tolist()
regiones = [x for x in regiones if str(x) != 'nan']
region = st.sidebar.multiselect(
    "Seleccione una o mas regiones", regiones, default=regiones)

if region != None:
    df = df[df["region"].isin(region)]
else:
    df = df

# with st.sidebar:
#     st.download_button("Descargar datos (csv)",
#                        convert_df(df), "maceda.csv",
#                        "text/csv", key="donwload-csv")
# --------------------------
# METRICAS
# --------------------------
eventos_totales = df["tipo de evento"].value_counts().tolist()
eventos_totales = sum(eventos_totales)

regiones_totales = df["region"].unique().tolist()
regiones_totales = len(regiones_totales)

actores_totales = df["actor principal"].unique().tolist()
actores_totales = len(actores_totales)

ataques = df[df["tipo de evento"] == "ATAQUE"]
ataques = ataques["tipo de evento"].value_counts().tolist()
ataques = sum(ataques)

protesta_violenta = df[df["detalle de evento"] == "PROTESTA VIOLENTA"]
protesta_violenta = protesta_violenta["tipo de evento"].value_counts().tolist()
protesta_violenta = sum(protesta_violenta)

pro_bar.progress(0.5, text="Construyendo gráficos...")

# # grafica tipos de eventos (tipo)
# eventos = df["tipo de evento"].unique().tolist()
# eventos_pie = go.Figure(data=[go.Pie(
#     labels=eventos, values=df["tipo de evento"].value_counts().tolist(), hole=.3)])

# # actores involucrados
# df_actor1 = df.groupby('actor principal').size().reset_index(name='cuenta')
# df_actor1.plot(kind='barh', x='actor principal', y='cuenta',
#                title='actor principal', figsize=(15, 8))

# # grafica regiones (linea)
# regiones = df["region"].unique().tolist()
# regiones_pie = go.Figure(data=[go.Pie(
#     labels=regiones, values=df["region"].value_counts().tolist(), hole=.3)])

# # grafica eventos
# tipos = df["tipo de evento"].unique().tolist()
# tipos_bar = go.Figure(data=[go.Pie(
#     labels=tipos, values=df["tipo de evento"].value_counts().tolist(), hole=.3)])

# eventos_tiempo = px.line(
#     df, x=df["fecha"], y=df["tipo de evento"], color=df["tipo de evento"])

# --------------------------
# MAIN
# --------------------------
st.subheader("Violencia etnopolítica - MACEDA")

pro_bar.progress(0.7, text="Construyendo gráficos...")

tabPanel, tabTable, tabBleau, tabInfo = st.tabs(
    ["Panel", "Tabla", "Informes", "Información"])

with tabPanel:  # graficos

    col1, col2, col3, col4, col5 = st.columns(5, gap="medium")
    col1.metric("Eventos", millify(eventos_totales, precision=2))
    col2.metric("Regiones", regiones_totales)
    col3.metric("Tipo de Actores", actores_totales)
    col4.metric(label="Ataques", value=ataques)
    col5.metric(label="Protestas violentas", value=protesta_violenta)

    style_metric_cards()

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        # ------------------------------------------------------------
        # TIPO DE EVENTOS
        # ------------------------------------------------------------
        fig = px.pie(df, values=df["tipo de evento"].value_counts().tolist(),
                     names=df["tipo de evento"].unique().tolist(),
                     title='Tipo de eventos',
                     color_discrete_sequence=px.colors.qualitative.Prism)
        fig.update_layout(title={
                          'xanchor': 'center', 'yanchor': 'top', 'y': .95, 'x': 0.5,
                          'font': {'size': 18}})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # ------------------------------------------------------------
        # REGIONES INVOLUCRADAS
        # ------------------------------------------------------------

        fig = px.pie(df, values=df["region"].value_counts().tolist(), names=df["region"].unique().tolist(),
                     title='Regiones involucradas',
                     color_discrete_sequence=px.colors.qualitative.Prism)
        fig.update_layout(title={
                          'xanchor': 'center', 'yanchor': 'top', 'y': .95, 'x': 0.5, 'font': {'size': 18}})
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------
    # ACTORES PARTICIPANTES
    # ------------------------------------------------------------
    st.divider()
    df_actor1 = df.groupby('actor principal').size().reset_index(name='cuenta')

    fig = px.bar(
        df_actor1,
        x="actor principal",
        y="cuenta",
        color="actor principal",
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig.update_layout(showlegend=False, xaxis_title='Tipo de actor',
                      title={'text': 'Actores participantes', 'xanchor': 'center', 'yanchor': 'top', 'y': .95, 'x': 0.5, 'font': {'size': 18}})
    st.plotly_chart(fig, use_container_width=True)

    # cantidad de conflictos por tipo de evento
    pro_bar.progress(0.8, text="Construyendo gráficos...")

    # ------------------------------------------------------------
    # DESARROLLO DE EVENTOS
    # ------------------------------------------------------------
    st.divider()
    serial = df.groupby('tipo de evento').size().reset_index(name='cuenta')
    fig = px.bar(
        serial,
        x="tipo de evento",
        y="cuenta",
        color="tipo de evento",
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig.update_layout(showlegend=False, xaxis_title='Tipo de evento',
                      title={'text': 'Desarrrollo de eventos', 'xanchor': 'center', 'yanchor': 'top', 'y': .95, 'x': 0.5, 'font': {'size': 18}})
    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------------------
    # DESARROLLO DE EVENTOS EN EL TIEMPO
    # ------------------------------------------------------------
    st.divider()

    serial = df.groupby(['fecha', 'tipo de evento']
                        ).size().reset_index(name='cuenta')
    serial['year'] = pd.DatetimeIndex(serial['fecha']).year
    serial['month'] = serial['fecha'].dt.strftime('%m')
    serial['mes'] = serial['fecha'].dt.strftime('%Y-%m')

    serial = serial.groupby(['mes', 'tipo de evento'],
                            as_index=False)['cuenta'].sum()

    figserial = px.line(serial, x="mes", y="cuenta", line_group="tipo de evento",
                        color="tipo de evento", markers=True,
                        color_discrete_sequence=px.colors.qualitative.Prism)
    figserial.update_layout(title={'text': 'Desarrollo de eventos en el tiempo',
                                   'xanchor': 'center', 'yanchor': 'top', 'y': .98, 'x': 0.5,
                                   'font': {'size': 18}},
                            xaxis_title='Fecha',
                            yaxis_title='No. de eventos',
                            showlegend=True,
                            legend=dict(
                                orientation="h",
                                title="",
                                yanchor="bottom",
                                y=1,
                                xanchor="right",
                                x=1,
                                traceorder="reversed",
                                title_font_family="Times New Roman",
                                font=dict(
                                    family="Courier",
                                    size=11,
                                    color="black"
                                ),
                                # bgcolor="LightSteelBlue",
                                # bordercolor="Black",
                                borderwidth=1
    ), height=600)

    st.plotly_chart(figserial, use_container_width=True)

    st.divider()

    # ------------------------------------------------------------
    # 4. CASOS POR REGIONES y ACTORES AFECTADOS
    # ------------------------------------------------------------

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        # REGIONES

        df_actor1 = df.groupby('region').size().reset_index(name='cuenta')

        fig = px.bar(
            df_actor1,
            x="region",
            y="cuenta",
            color="region",
            title="Casos por regiones",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig.update_layout(showlegend=False, xaxis_title=None,
                          yaxis_title=None, height=500,
                          title={
                              'xanchor': 'center', 'yanchor': 'top', 'y': .95, 'x': 0.5, 'font': {'size': 18}})
        st.plotly_chart(fig, use_container_width=True)

        st.write("Como es esperable, los eventos son más frecuentes en el conflicto mapuche-estado chileno, dentro de la región de la Araucanía, donde se concentra la mayor cantidad de eventos. Donde el 99,5% de la población indigena pertenece a la etnia mapuche, representando el 23,4% de la población total de esta región. ")
        st.markdown("""
                    El análisis de frecuencia de eventos por región muestra que la mayoría de los eventos se concentran en las siguientes regiones:

1. Araucanía: 2760 eventos.
2. Bio-Bio: 1143 eventos.
3. Metropolitana de Santiago: 261 eventos.

Otras regiones como Los Ríos y Los Lagos también tienen una cantidad significativa de eventos, mientras que regiones como Ñuble y Aysén del General Carlos Ibáñez del Campo tienen muy pocos eventos registrados.""")
        st.markdown("""
Interpretación
1. Concentración de eventos: La gran mayoría de los eventos se concentran en las regiones de Araucanía y Bio-Bio. Esto podría indicar un foco de actividades específicas en estas áreas.
2. Regiones con menos eventos: Algunas regiones tienen muy pocos eventos registrados, lo cual podría ser debido a menor actividad o menor cobertura de registro de eventos en esas áreas.""")
        st.markdown("Fuente: [https://oes.ufro.cl/index.php/oes-ufro/estudios-regionales](https://oes.ufro.cl/index.php/oes-ufro/estudios-regionales/estudios-externos/category/12-estudiosexternos#:~:text=En%20la%20regi%C3%B3n%20de%20La%20Araucan%C3%ADa%2C%20del%20total%20de%20ind%C3%ADgenas,poblaci%C3%B3n%20total%20de%20esta%20regi%C3%B3n).")

    with col2:
        # ACTOR AFECTADO

        df_afectado = df.groupby(
            'actor afectado').size().reset_index(name='cuenta')

        fig = px.bar(
            df_afectado,
            x="actor afectado",
            y="cuenta",
            color="actor afectado",
            title="Actores afectados",
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        fig.update_layout(showlegend=False, xaxis_title=None,
                          yaxis_title=None, height=500,
                          title={
                              'xanchor': 'center', 'yanchor': 'top', 'y': .95, 'x': 0.5, 'font': {'size': 18}})
        st.plotly_chart(fig, use_container_width=True)

        st.write("En cuanto a los actores afectados, se observa que la mayor cantidad de eventos afectan a la población civil, seguido por la policía y los actores no estatales. ")

    st.divider()

    st.markdown("""##### Análisis Temporal de Eventos por Región""")
    # Agrupar los datos por región y año para realizar el análisis temporal
    eventos_temporal_region = df.groupby(
        ['region', 'año']).size().unstack(fill_value=0)

    # Crear un gráfico de líneas para mostrar el número de eventos por región a lo largo del tiempo
    plt.figure(figsize=(14, 10))

    for region in eventos_temporal_region.index:
        plt.plot(eventos_temporal_region.columns,
                 eventos_temporal_region.loc[region], label=region)

    plt.xlabel('Año')
    plt.ylabel('Número de Eventos')
    plt.title('Número de Eventos por Región a lo Largo del Tiempo')
    plt.legend(title='Región', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    # plt.show()
    st.pyplot(plt, use_container_width=True)

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("""El análisis temporal revela cómo ha cambiado el número de eventos en cada región a lo largo del tiempo.

Observaciones Clave:

1. Araucanía:

Ha experimentado un aumento significativo en el número de eventos desde la década de 1990 hasta 2021.
El incremento es particularmente notable a partir de 2009, con un pico creciente en la última década.

2. Bio-Bio:

También muestra un aumento en la frecuencia de eventos, aunque de manera más gradual comparado con Araucanía.
Desde 2014, hay un crecimiento constante hasta 2021.

3. Regiones con menos eventos:

Otras regiones como Los Ríos y Los Lagos muestran eventos más esporádicos y menos frecuentes.
Regiones como Arica y Parinacota, Aysén y Antofagasta tienen muy pocos o ningún evento registrado en la mayoría de los años.""")
    with col2:
        st.markdown("""
    Interpretación
1. Crecimiento Sostenido en Araucanía y Bio-Bio:

El notable aumento en las regiones de Araucanía y Bio-Bio sugiere una escalada de actividades específicas en estas áreas. Puede ser útil investigar los factores subyacentes que han contribuido a este aumento.

2. Estabilidad en Otras Regiones:

La estabilidad o baja frecuencia de eventos en otras regiones puede reflejar menor actividad o distintos contextos socioeconómicos y políticos.
                """)

    # ------------------------------------------------------------
    # CASOS POR COMUNA
    # ------------------------------------------------------------
    st.divider()
    st.markdown("##### Casos por comuna")
    with st.expander("Umbral de casos"):
        casos = st.slider(
            "Seleccione cantidad de casos mínimos a considerar", 10, 1000, 50)

    df = df[df.groupby('comuna').comuna.transform('size') > casos]
    df_comuna = df.groupby('comuna').size().reset_index(name='casos')
    fig = px.bar(
        df_comuna,
        x="comuna",
        y="casos",
        color="comuna",
        title=None,
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    st.write("La comuna de Ercilla, en la región de la Araucanía, es la que presenta la mayor cantidad de eventos, seguida por la comuna de Collipulli. ")

    st.divider()

    # ------------------------------------------------------------
    # FACTORES CONTRIBUTIVOS
    # ------------------------------------------------------------

    st.markdown("##### Factores Contributivos")
    st.markdown("""Para comprender mejor los factores que contribuyen al aumento de eventos en la región de Araucanía, es importante considerar los siguientes aspectos:Explorar qué factores específicos están impulsando el aumento en el número de eventos en Araucanía y Bio-Bio.
                
Comparar los tipos de eventos que ocurren en las regiones con más eventos frente a las regiones con menos eventos.""")

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        araucania_data = df[df['region'] == 'ARAUCANIA']

        # Análisis de la frecuencia de tipos de eventos en Araucanía
        eventos_tipo_frecuencia = araucania_data['tipo de evento'].value_counts(
        ).reset_index()
        eventos_tipo_frecuencia.columns = ['tipo de evento', 'num_eventos']

        # Gráfico de barras de frecuencia de tipos de eventos
        plt.figure(figsize=(14, 8))
        plt.bar(eventos_tipo_frecuencia['tipo de evento'],
                eventos_tipo_frecuencia['num_eventos'])
        plt.xticks(rotation=90)
        plt.xlabel('Tipo de Evento', fontsize=16)
        plt.ylabel('Número de Eventos', fontsize=16)
        plt.title('Frecuencia de Tipos de Eventos en Araucanía', fontsize=18)
        st.pyplot(plt, use_container_width=True)

        st.markdown("""El análisis de la frecuencia de tipos de eventos en Araucanía revela que los ataques son el tipo de evento más común, seguido por las protestas violentas y las ocupaciones de tierras. Estos hallazgos sugieren que los conflictos en la región están relacionados con una variedad de actividades, incluyendo ataques, protestas y ocupaciones de tierras.""")

    with col2:
        # ACTORES
        # Análisis de la frecuencia de actores involucrados en Araucanía
        df = df[df['region'] == 'ARAUCANIA']
        actores_tipo_frecuencia = df['tipo de actor'].value_counts(
        ).reset_index()
        actores_tipo_frecuencia.columns = ['tipo de actor', 'num_eventos']

        # Gráfico de barras de frecuencia de actores involucrados
        plt.figure(figsize=(14, 8))
        plt.bar(actores_tipo_frecuencia['tipo de actor'],
                actores_tipo_frecuencia['num_eventos'])
        plt.xticks(rotation=90)
        plt.xlabel('Tipo de Actor', fontsize=16)
        plt.ylabel('Número de Eventos', fontsize=16)
        plt.title(
            'Frecuencia de Tipos de Actores Involucrados en Araucanía', fontsize=18)
        st.pyplot(plt, use_container_width=True)

        st.markdown("""El análisis de la frecuencia de actores involucrados en Araucanía muestra que los actores no estatales son los más comunes, seguidos por la policía y la población civil. Estos hallazgos sugieren que los conflictos en la región involucran a una variedad de actores.
1. Predominio de Actores Mapuche: Los actores relacionados con la comunidad Mapuche (Otros Mapuche y Organización Mapuche) son predominantes en los eventos de Araucanía.
2. Participación de la Comunidad: La comunidad en general también tiene una participación significativa en los eventos.""")

    st.divider()

    # ------------------------------------------------------------
    # NUBE DE PALABRAS
    # ------------------------------------------------------------

    st.markdown("##### Análisis de Descripción de Eventos")
    from wordcloud import WordCloud

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        descriptions = df['descripcion'].dropna().tolist()

        text = ' '.join(descriptions)

        # Remove articles and spaces
        text = re.sub(r'\b(el|la|los|las|al|lo|le|un|una|unos|unas|a|ante|bajo|cabe|con|contra|de|desde|en|entre|hacia|hasta|para|por|según|sin|so|sobre|tras|que|se|del|su|fue|y|tres|sus)\b',
                      '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text)

        # Create a word cloud
        wordcloud = WordCloud(width=900, height=800,
                              background_color='white').generate(text)

        # Plot the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')

        st.pyplot(plt, use_container_width=True)

with col2:
    st.markdown("""Para obtener una comprensión más profunda, revisamos las descripciones de los eventos para identificar patrones o temas recurrentes utilizando técnicas de procesamiento de lenguaje natural (NLP).""")
    st.markdown(""" **Nube de Palabras desde los eventos en Araucanía**

La nube de palabras generada a partir de las descripciones de los eventos en Araucanía resalta las palabras más frecuentes, proporcionando una visión rápida de los temas recurrentes. Algunas de las palabras más prominentes incluyen:

- "Mapuche": Refleja la fuerte presencia de actores relacionados con la comunidad Mapuche.
- "Ataque": Indica la alta frecuencia de eventos clasificados como ataques.
- "Protesta": Corrobora la presencia significativa de protestas en la región.
""")

    st.markdown("""
    **Conclusión**

    Los factores que contribuyen al aumento de eventos en la región de Araucanía parecen estar fuertemente relacionados con conflictos y actividades de la comunidad Mapuche, incluyendo ataques, protestas y ocupaciones de tierras.""")


pro_bar.progress(0.9, text="Construyendo gráficos...")
# --------------------------
with tabTable:  # tabla de datos
    # --------------------------
    df['fecha'] = df['fecha'].dt.date
    st.dataframe(df, height=600)


# --------------------------
with tabBleau:  # graficos personalizados
    # @ st.cache_resource
    # def get_pyg_renderer() -> "StreamlitRenderer":
    #     return StreamlitRenderer(df, spec_io_mode="rw")

    # renderer = get_pyg_renderer()
    # renderer.explorer()

    pyg_app = StreamlitRenderer(df, spec_io_mode="rw")
    pyg_app.explorer()


# --------------------------
with tabInfo:
    col, cor = st.columns(2, gap="medium")
    with col:
        st.write("El Proyecto de Datos Mapuche (MDP) tiene como objetivos identificar, digitalizar, compilar, procesar y armonizar información cuantitativa respecto al pueblo Mapuche. ")

        st.write("Basados en MACEDA (Mapuche Chilean State Conflict Event Database), primer registro sistemático de eventos relacionados al conflicto entre el pueblo mapuche y el estado chileno.\n")

        st.write("MPD reporta información del conflicto entre el Estado Chileno y el pueblo mapuche. La Base de Datos de Eventos sobre el Conflicto Mapuche-Estado Chileno MACEDA (por su acrónimo en inglés) reporta más de 4500 eventos para el período 1990-2021.")
        st.write("")
        st.markdown(
            "[Descargar datos (csv 2.2MB)](https://sites.google.com/view/danyjaimovich/links/mdp)")
        st.markdown(
            "[Documentación (377KB)](https://data.vgclab.cl/public_data/mdp_conflicto_maceda_codigos.pdf)")
        st.write("")
        st.write("Cómo citar:")
        st.write("Cayul, P., A. Corvalan, D. Jaimovich, and M. Pazzona (2022). Introducing MACEDA: New Micro-Data on an Indigenous Self-Determination Conflict. Journal of Peace Research ")

    with cor:
        st.image("images/maceda.jpg", width=500)

pro_bar.empty()
