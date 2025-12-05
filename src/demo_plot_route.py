#src\demo_plot_route.py
"""
Script de demo: carga/descarga grafo, selecciona un par y dibuja la ruta A*.
Guarda la imagen en `datos/demo_route.png`.
"""
import os
import sys

try:
    import osmnx as ox
    from src import problemaRuta as pr
except Exception as e:
    print("Error importando dependencias:", e)
    raise

DATA_DIR = "datos"
os.makedirs(DATA_DIR, exist_ok=True)
GRAPH_PATH = os.path.join(DATA_DIR, "grafo.graphml")
PLACE = "Tec de Monterrey campus Guadalajara, Zapopan, México"

try:
    if os.path.exists(GRAPH_PATH):
        print("Cargando grafo desde:", GRAPH_PATH)
        G = ox.load_graphml(GRAPH_PATH)
    else:
        print("Descargando grafo (puede tardar)...")
        G = ox.graph_from_address(PLACE, dist=3000, network_type="drive")
        ox.save_graphml(G, GRAPH_PATH)

    print("Nodos en grafo:", len(G.nodes))

    # reproyectado para selección de pares
    G_proj = ox.project_graph(G)

    pairs = pr.pick_pairs_by_distance(G, G_proj, n_pairs_each=1)
    pair = None
    for k, v in pairs.items():
        if v:
            pair = v[0]
            break
    if pair is None:
        print("No se encontraron parejas en el grafo.")
        sys.exit(1)

    u, v, dist = pair
    print(f"Pair selected: {u} -> {v} (approx {dist:.1f} m)")

    # Ejecutar A* directamente sobre GraphSearchProblem para evitar
    # cálculo oneroso/posible error de path_length dentro de run_search.
    from simpleai.search import astar
    problem = pr.GraphSearchProblem(G, initial=u, goal=v, projected_graph=G_proj)
    result = astar(problem)
    # normalizar path
    try:
        path_states = result.path()
        if len(path_states) > 0 and isinstance(path_states[0], tuple):
            path = [s if isinstance(s, int) else s[0] for s in path_states]
        else:
            path = [int(s) for s in path_states]
    except Exception:
        try:
            path = [int(result.state)]
        except Exception:
            path = None

    print("Search done. path length:", 0 if path is None else len(path))
    print("path contents:", path)
    # defensiva: filtrar posibles None en la ruta
    if path:
        path = [int(p) for p in path if p is not None]
    print("path filtered:", path)
    if not path:
        print("No se obtuvo ruta.")
        sys.exit(1)

    out_path = os.path.join(DATA_DIR, "demo_route.png")
    ok = pr.plot_route_on_map(G, path, out_path=out_path, show=False)
    print("Plot saved:", ok, out_path)

except Exception as exc:
    print("Demo failed:", exc)
    raise
