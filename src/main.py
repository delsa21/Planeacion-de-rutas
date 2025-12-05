import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import sys
import osmnx as ox
from simpleai.search import astar

# --- CORRECCIÓN DE RUTAS ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importamos tus módulos
from src.kdtree import load_graph, extract_nodes_coords, run_tests_kdtree
from src.testRutas import run_routing_tests
from src.voronoi import GestorHospitales
from src.problemaRuta import crear_problema

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Rutas Rápidas", layout="centered")

# --- TÍTULO ---
st.title("🗺️ Proyecto de Rutas (Versión Rápida)")
st.write("Sistema optimizado para demostraciones ágiles.")

# --- PASO 1: CARGAR EL MAPA ---
st.header("1. Cargar el Mapa")

if 'mapa_listo' not in st.session_state:
    st.session_state['mapa_listo'] = False

if st.button("📍 Cargar Mapa de Zapopan (1km)"):
    with st.spinner("Descargando mapa pequeño..."):
        # OPTIMIZACIÓN CLAVE: dist=1000 (1km) para que sea rápido
        place = "Tec de Monterrey campus Guadalajara, Zapopan, Jalisco, México"
        G = load_graph(place, dist=1000) 
        nodes, coords = extract_nodes_coords(G)
        
        st.session_state['grafo'] = G
        st.session_state['nodos'] = nodes
        st.session_state['coords'] = coords
        st.session_state['mapa_listo'] = True
    
    st.success(f"¡Mapa cargado! ({len(nodes)} nodos)")

# --- MENÚ PRINCIPAL ---
if st.session_state['mapa_listo']:
    
    G = st.session_state['grafo']
    nodes = st.session_state['nodos']
    coords = st.session_state['coords']

    st.divider()
    opcion = st.selectbox(
        "Selecciona una actividad:", 
        ["Selecciona...", "Pruebas KD-Tree", "Comparar Algoritmos", "Ver Voronoi", "Ver Ruta Visual"]
    )

    # --- OPCIÓN A: KD-TREE ---
    if opcion == "Pruebas KD-Tree":
        st.subheader("🌳 Rendimiento de Búsqueda")
        if st.button("Ejecutar Test"):
            run_tests_kdtree(G, nodes, coords, n_tests=20, out_csv="datos/kdtree_fast.csv")
            df = pd.read_csv("datos/kdtree_fast.csv")
            st.dataframe(df)
            st.line_chart(df[['tiempo_kdtree_seg', 'tiempo_bruteforce_seg']])

    # --- OPCIÓN B: COMPARAR ALGORITMOS ---
    elif opcion == "Comparar Algoritmos":
        st.subheader("📊 Comparativa (BFS, DFS, A*, UCS)")
        st.info("Nota: Se probarán rutas cortas para evitar demoras.")
        
        if st.button("Calcular Rutas"):
            with st.spinner("Procesando..."):
                # Ejecuta la versión optimizada de testRutas
                run_routing_tests(G, num_pairs=3, out_csv="datos/rutas_fast.csv")
            
            try:
                df = pd.read_csv("datos/rutas_fast.csv")
                st.dataframe(df)
            except:
                st.error("No se generaron datos. Verifica que testRutas.py esté actualizado.")

    # --- OPCIÓN C: VORONOI ---
    elif opcion == "Ver Voronoi":
        st.subheader("🏥 Hospitales (Voronoi)")
        n_hosp = st.slider("Cantidad de Hospitales", 3, 8, 5)
        
        if st.button("Generar Diagrama"):
            gh = GestorHospitales(G)
            gh.cargar_hospitales_ficticios(n=n_hosp)
            
            # Emergencia simulada
            t_lat, t_lon = coords[0]
            hosp, dist = gh.encontrar_hospital_cercano(t_lat, t_lon)
            st.success(f"Ir a: {hosp['nombre']} ({dist:.0f}m)")
            
            # Gráfica
            import numpy as np
            from scipy.spatial import Voronoi, voronoi_plot_2d
            points = np.array(gh.coords_hosp)
            vor = Voronoi(points)
            fig, ax = plt.subplots()
            voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='orange')
            ax.plot(points[:, 0], points[:, 1], 'ko')
            st.pyplot(fig)

    # --- OPCIÓN D: RUTA VISUAL ---
    elif opcion == "Ver Ruta Visual":
        st.subheader("📍 Ruta A* en Mapa")
        if st.button("Generar Ruta Aleatoria"):
            origen = random.choice(nodes)
            destino = random.choice(nodes)
            
            problema = crear_problema(G, origen, destino)
            resultado = astar(problema, graph_search=True)
            
            if resultado:
                st.success(f"Costo: {resultado.cost:.1f}m")
                camino = [s for a, s in resultado.path()]
                fig, ax = ox.plot_graph_route(G, camino, node_size=0, bgcolor='white', show=False, close=False)
                st.pyplot(fig)
            else:
                st.warning("Puntos desconectados. Intenta de nuevo.")

else:
    st.info("👆 Carga el mapa para empezar.")