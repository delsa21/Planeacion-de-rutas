# src\main.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import sys
import osmnx as ox
from simpleai.search import astar

# ajustar path para importar módulos de src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# importar módulos del proyecto
from src.kdtree import load_graph, extract_nodes_coords, run_tests_kdtree
from src.testRutas import run_routing_tests
from src.voronoi import GestorHospitales
from src.problemaRuta import crear_problema

# configuración de la interfaz (del main antiguo)
st.set_page_config(page_title="Planeación de rutas", layout="centered")
st.title(
    "Proyecto: Implementación de una plataforma de planeación de rutas a partir de algoritmos de búsqueda"
)
st.header("1. Cargar el mapa")

# ubicación por defecto
PLACE_DEFAULT = "Tec de Monterrey campus Guadalajara, Zapopan, Jalisco, México"

# estado inicial
if "mapa_listo" not in st.session_state:
    st.session_state["mapa_listo"] = False

# botón para cargar mapa
if st.button("Cargar mapa de Zapopan"):
    with st.spinner("Descargando mapa..."):
        G = load_graph(PLACE_DEFAULT, dist=5000)
        nodes, coords = extract_nodes_coords(G)

        st.session_state["grafo"] = G
        st.session_state["nodos"] = nodes
        st.session_state["coords"] = coords
        st.session_state["mapa_listo"] = True

    st.success(f"Mapa cargado correctamente ({len(nodes)} nodos)")

# menú
if st.session_state["mapa_listo"]:

    G = st.session_state["grafo"]
    nodes = st.session_state["nodos"]
    coords = st.session_state["coords"]

    st.divider()
    opcion = st.selectbox(
        "Selecciona una actividad:",
        [
            "Selecciona...",
            "Pruebas KD-Tree",
            "Comparar algoritmos",
            "Servicio de emergencias (Voronoi)",
            "Ver ruta visual",
        ],
    )

    # módulo kd-tree
    if opcion == "Pruebas KD-Tree":
        st.subheader("Rendimiento de búsqueda (KD-Tree vs fuerza bruta)")
        st.write("Compara el tiempo de búsqueda del nodo más cercano a una coordenada.")

        if st.button("Ejecutar pruebas de árbol KD"):
            run_tests_kdtree(G, n_tests=20, out_csv="datos/kdtree_fast.csv")

            try:
                df = pd.read_csv("datos/kdtree_fast.csv")
                st.dataframe(df)

                cols_tiempo = [c for c in df.columns if "time" in c]
                if len(cols_tiempo) >= 2:
                    st.line_chart(df[cols_tiempo])

            except Exception as e:
                st.error(f"Error leyendo resultados: {e}")

    # módulo para comparar algoritmos
    elif opcion == "Comparar algoritmos":
        st.subheader("Comparativa (BFS, DFS, A*, UCS, IDDFS)")
        st.write("Evalúa diferentes estrategias de búsqueda entre pares de nodos.")

        if st.button("Calcular rutas"):
            with st.spinner("Procesando algoritmos..."):
                run_routing_tests(G, num_pairs=3, out_csv="datos/rutas_fast.csv")

                try:
                    df = pd.read_csv("datos/rutas_fast.csv")
                    st.dataframe(df)
                except:
                    st.error("No se generaron datos.")

    # módulo voronoi
    elif opcion == "Servicio de emergencias (Voronoi)":
        st.subheader("Hospitales cercanos")
        st.markdown(
            """
            Este módulo busca hospitales reales en OSM y divide el mapa en regiones de Voronoi.
            Cada punto del territorio queda asignado al hospital más cercano.
            """
        )

        usar_reales = st.checkbox(
            "Buscar hospitales reales en OpenStreetMap", value=True
        )

        if st.button("Generar diagrama de Voronoi"):
            gh = GestorHospitales(G)

            if usar_reales:
                with st.spinner("Descargando hospitales reales..."):
                    gh.cargar_hospitales_reales(PLACE_DEFAULT, dist=4500)
            else:
                gh.cargar_hospitales_ficticios(n=5)

            if len(gh.hospitales) > 0:
                idx = random.randint(0, len(coords) - 1)
                t_lat, t_lon = coords[idx]

                hosp, dist = gh.encontrar_hospital_cercano(t_lat, t_lon)

                col1, col2 = st.columns(2)
                with col1:
                    st.error(
                        f"Emergencia reportada en:\nLat: {t_lat:.4f}, Lon: {t_lon:.4f}"
                    )
                with col2:
                    st.success(
                        f"Ir al hospital:\n**{hosp['nombre']}**\nDistancia: {dist:.0f} metros"
                    )

                fig = gh.visualizar_voronoi()
                if fig:
                    st.pyplot(fig)
            else:
                st.warning("No se encontraron hospitales para generar el diagrama.")

    # módulo ruta visual
    elif opcion == "Ver ruta visual":
        st.subheader("Visualizador de rutas (A*)")

        if st.button("Generar ruta aleatoria"):
            origen = random.choice(nodes)
            destino = random.choice(nodes)

            st.write(f"Calculando ruta de nodo {origen} a {destino}...")
            problema = crear_problema(G, origen, destino)
            resultado = astar(problema, graph_search=True)

            if resultado:
                st.success(f"Ruta encontrada, costo: {resultado.cost:.1f} m")
                camino = [s for a, s in resultado.path()]

                fig, ax = ox.plot_graph_route(
                    G,
                    camino,
                    node_size=0,
                    bgcolor="white",
                    show=False,
                    close=False,
                )
                st.pyplot(fig)
            else:
                st.warning("Puntos desconectados o inalcanzables. Intenta de nuevo.")

else:
    st.info("Por favor, carga el mapa para empezar.")
