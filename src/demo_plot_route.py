# src/demo_plot_route.py
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
    # carga o descarga del grafo
    if os.path.exists(GRAPH_PATH):
        print("Cargando grafo desde archivo local...")
        G = ox.load_graphml(GRAPH_PATH)
    else:
        print("Descargando grafo, esto puede tardar...")
        G = ox.graph_from_address(PLACE, dist=3000, network_type="drive")
        ox.save_graphml(G, GRAPH_PATH)

    print("Nodos cargados:", len(G.nodes))

    # reproyección para cálculos geométricos rápidos
    G_proj = ox.project_graph(G)

    # selección de pares de prueba por distancia
    pairs = pr.pick_pairs_by_distance(G, G_proj, n_pairs_each=1)

    pair = None
    for _, vals in pairs.items():
        if vals:
            pair = vals[0]
            break

    if pair is None:
        print("No se encontraron pares adecuados.")
        sys.exit(1)

    u, v, dist = pair
    print(f"Par seleccionado: {u} -> {v} (~{dist:.1f} m)")

    # ejecución directa de a* sobre el problema base
    from simpleai.search import astar

    problem = pr.GraphSearchProblem(
        G,
        initial=u,
        goal=v,
        projected_graph=G_proj,
    )

    result = astar(problem)

    # extracción defensiva del camino
    try:
        raw_path = result.path()
        if raw_path and isinstance(raw_path[0], tuple):
            path = [s if isinstance(s, int) else s[0] for s in raw_path]
        else:
            path = [int(s) for s in raw_path]
    except Exception:
        try:
            path = [int(result.state)]
        except Exception:
            path = None

    print("Longitud de la ruta:", 0 if path is None else len(path))
    print("Ruta:", path)

    # eliminación de posibles valores nulos
    if path:
        path = [p for p in path if p is not None]

    if not path:
        print("No se pudo recuperar una ruta válida.")
        sys.exit(1)

    # generación de la figura final
    out_path = os.path.join(DATA_DIR, "demo_route.png")
    ok = pr.plot_route_on_map(G, path, out_path=out_path, show=False)

    print("Imagen generada:", ok)
    print("Archivo guardado en:", out_path)

except Exception as exc:
    print("Error en demo:", exc)
    raise
