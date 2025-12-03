# src/kdtree.py
"""
Módulo: kdtree.py
Responsable de:
 - Descargar grafo desde OSMnx.
 - Extraer nodos y coordenadas (lat, lon).
 - Construir KDTree para búsqueda optimizada.
 - Comparar KDTree vs búsqueda exhaustiva.
 - Generar pruebas y mediciones para el reporte.
"""

import osmnx as ox
import numpy as np
from scipy.spatial import KDTree

from src.utils import haversine_meters, timeit, ensure_dir, save_csv, format_seconds


# ---------------------------------------------
# 1) Cargar grafo desde OSMnx
# ---------------------------------------------
def load_graph(place, dist=10000, network_type="drive"):
    """
    Descarga el grafo alrededor de 'place' en un radio 'dist' metros.
    """
    print(f"Descargando grafo de: {place}")
    G = ox.graph_from_address(place, dist=dist, network_type=network_type)
    print(f"Grafo cargado: {len(G.nodes)} nodos, {len(G.edges)} aristas")
    return G


# ---------------------------------------------
# 2) Extraer nodos y coordenadas
# ---------------------------------------------
def extract_nodes_coords(G):
    """
    Regresa dos listas:
      - nodes: IDs de cada nodo
      - coords: lista de tuplas (lat, lon)
    """
    nodes = []
    coords = []
    for nid, data in G.nodes(data=True):
        nodes.append(nid)
        coords.append((data["y"], data["x"]))
    return nodes, coords


# ---------------------------------------------
# 3) Construir KDTree
# ---------------------------------------------
@timeit
def build_kdtree(coords):
    """
    Construye el KDTree a partir de las coordenadas.
    Regresa: (KDTree, tiempo)
    """
    arr = np.array(coords)
    tree = KDTree(arr)
    return tree


# ---------------------------------------------
# 4) Búsqueda con KDTree
# ---------------------------------------------
def nearest_kdtree(tree, coords, nodes, query_point):
    """
    Busca el nodo más cercano a 'query_point' (lat, lon) usando KDTree.
    Regresa: (node_id, distancia_metros_approx)
    """
    dist, idx = tree.query(query_point)
    node_id = nodes[idx]
    dist_m = haversine_meters(query_point, coords[idx])
    return node_id, dist_m


# ---------------------------------------------
# 5) Búsqueda exhaustiva (sin KD-Tree)
# ---------------------------------------------
def nearest_bruteforce(coords, nodes, query_point):
    """
    Búsqueda lineal sobre todas las coordenadas.
    Muy lenta para grafos grandes (propósito: comparación).
    """
    best_idx = None
    best_dist = float("inf")

    for i, pt in enumerate(coords):
        d = haversine_meters(query_point, pt)
        if d < best_dist:
            best_dist = d
            best_idx = i

    return nodes[best_idx], best_dist


# ---------------------------------------------
# 6) Generar pruebas para el reporte
# ---------------------------------------------
def run_tests_kdtree(G, nodes, coords, n_tests=20, out_csv="datos/kdtree_tests.csv"):
    """
    Ejecuta 20 pruebas:
    - 20 puntos tomados de coords (los primeros 20 o aleatorios)
    - mide tiempos KDTree vs brute force
    - guarda resultados en CSV
    """
    print("\nEjecutando pruebas KDTree...")

    # Construir KDTree
    (tree, t_build) = build_kdtree(coords)
    print(f"KD-Tree construido en: {format_seconds(t_build)}")

    tests = []
    sample_points = coords[:n_tests]  # puedes cambiar por aleatoria si quieres

    for i, q in enumerate(sample_points):
        # KDTree
        (_, t_kdt) = nearest_kdtree_timed(tree, coords, nodes, q)

        # Brute force
        (_, t_bf) = nearest_bruteforce_timed(coords, nodes, q)

        tests.append(
            {
                "test_id": i + 1,
                "query_lat": q[0],
                "query_lon": q[1],
                "tiempo_kdtree_seg": t_kdt,
                "tiempo_bruteforce_seg": t_bf,
            }
        )

        print(
            f"Test {i+1}: KDTree={format_seconds(t_kdt)} | Brute={format_seconds(t_bf)}"
        )

    ensure_dir("datos")
    save_csv(tests, out_csv)
    print(f"\nResultados guardados en: {out_csv}")


# ---------------------------------------------
# 7) Wrapper para medir tiempos
# ---------------------------------------------
@timeit
def nearest_kdtree_timed(tree, coords, nodes, query_point):
    return nearest_kdtree(tree, coords, nodes, query_point)


@timeit
def nearest_bruteforce_timed(coords, nodes, query_point):
    return nearest_bruteforce(coords, nodes, query_point)


if __name__ == "__main__":
    place = "Tec de Monterrey campus Guadalajara, Zapopan, Jalisco, México"
    G = load_graph(place)

    nodes, coords = extract_nodes_coords(G)

    run_tests_kdtree(G, nodes, coords, n_tests=10)  # prueba rápida de 10
