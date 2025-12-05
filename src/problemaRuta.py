# src/problemaRuta.py
"""
Planeador de rutas usando SimpleAI (BFS, DFS, UCS, IDDFS, A*).

Uso:
    python -m src.problemaRuta  # muestra ayuda mínima y demo

Características:
 - Implementa wrapper para SimpleAI SearchProblem sobre un grafo NetworkX (OSMnx).
 - Mide tiempos, nodos expandidos (estimado) y longitud de la ruta.
 - Soporta grafos proyectados (usa 'x','y' para heurística euclidiana) o lat/lon (haversine).
 - Exporta resultados a CSV mediante src.utils.save_csv.
"""

from __future__ import annotations
import time
import os
import argparse
from typing import Any, List, Tuple, Optional, Dict

import networkx as nx
from simpleai.search import SearchProblem, breadth_first, depth_first, uniform_cost, astar

from src.utils import (
    haversine_meters,
    timeit,
    ensure_dir,
    save_csv,
    format_seconds,
    validate_node_in_graph,
)

# Default data dir
DATA_DIR = "datos"
ensure_dir(DATA_DIR)


def plot_route_on_map(G, path, out_path: Optional[str] = None, show: bool = False):
    """
    Dibuja y opcionalmente guarda una imagen de la ruta `path` sobre el grafo `G`.
    - G: grafo (NetworkX / OSMnx)
    - path: lista de node ids que forman la ruta
    - out_path: si se da, guarda la imagen en esa ruta (PNG). Si None, no guarda.
    - show: si True, muestra la figura con matplotlib.

    Devuelve True si se generó la imagen o se mostró correctamente, False en caso de fallo.
    """
    try:
        import osmnx as ox
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot_route_on_map] error importando librerías de visualización: {e}")
        return False

    if not path:
        print("[plot_route_on_map] path vacío, nada para graficar.")
        return False

    try:
        fig, ax = ox.plot_graph_route(
            G,
            path,
            route_color="red",
            route_linewidth=3,
            node_size=0,
            show=False,
            close=False,
        )
    except Exception as e:
        print(f"[plot_route_on_map] error al graficar ruta: {repr(e)}")
        return False

    if out_path:
        ensure_dir(os.path.dirname(out_path) or ".")
        try:
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"[plot_route_on_map] imagen guardada en: {out_path}")
        except Exception as e:
            print(f"[plot_route_on_map] no se pudo guardar la imagen: {e}")

    if show:
        try:
            plt.show()
        except Exception as e:
            print(f"[plot_route_on_map] error mostrando figura: {e}")

    try:
        plt.close(fig)
    except Exception:
        pass

    return True


# ------------------------------
# Helper: distancia entre nodos
# ------------------------------
def node_distance_m(G_proj: nx.Graph, u: int, v: int) -> float:
    """
    Distancia aproximada entre nodos u y v en metros.
    Si G_proj parece proyectado (tiene 'x' y 'y') usa distancia euclidiana.
    Si no, usa haversine sobre lat/lon almacenados en 'y'/'x' (lat, lon).
    """
    du = G_proj.nodes[u]
    dv = G_proj.nodes[v]
    # detect projected coords presence
    if ("x" in du and "y" in du) and ("x" in dv and "y" in dv):
        dx = du["x"] - dv["x"]
        dy = du["y"] - dv["y"]
        return (dx * dx + dy * dy) ** 0.5
    # fallback: haversine expects (lat,lon)
    a = (du.get("y", du.get("lat")), du.get("x", du.get("lon")))
    b = (dv.get("y", dv.get("lat")), dv.get("x", dv.get("lon")))
    return haversine_meters(a, b)


# =======================================================
#  IDDFS (Iterative Deepening Depth-First Search) local
# =======================================================
def depth_limited_search(problem, limit: int):
    from collections import deque

    stack = deque()
    stack.append((problem.initial_state, 0, [problem.initial_state]))

    cutoff_occurred = False

    while stack:
        state, depth, path = stack.pop()

        if problem.is_goal(state):
            return ("success", path)

        if depth == limit:
            cutoff_occurred = True
            continue

        for action in problem.actions(state):
            child = problem.result(state, action)
            if child not in path:
                stack.append((child, depth + 1, path + [child]))

    if cutoff_occurred:
        return ("cutoff", None)

    return ("failure", None)


def iterative_deepening(problem, max_depth: int = 2000):
    class Result:
        def __init__(self, path):
            self._path = path

        def path(self):
            return self._path

    for limit in range(max_depth):
        result, path = depth_limited_search(problem, limit)
        if result == "success":
            return Result(path)
        elif result == "failure":
            return None

    return None


# ------------------------------
# SimpleAI SearchProblem para grafos
# ------------------------------
class GraphSearchProblem(SearchProblem):
    """
    Problema para SimpleAI que modela moverse por nodos del grafo.
    El estado es el node id actual (entero).
    Actions: vecinos a los que se puede ir.
    Cost: peso de la arista (atributo 'length' preferido).
    """

    def __init__(self, G: nx.Graph, initial: int, goal: int, projected_graph: Optional[nx.Graph] = None):
        """
        G: grafo networkx tal como lo produce OSMnx (dirigido o no).
        initial: nodo inicio (id)
        goal: nodo objetivo (id)
        projected_graph: si el grafo reproyectado (para heurística), pasar G_proj; puede ser el mismo que G.
        """
        super(GraphSearchProblem, self).__init__(initial_state=initial)
        self.G = G
        self.initial = initial
        self.goal = goal
        self.G_proj = projected_graph or G

        # Contadores para métricas: incrementamos cuando actions() es llamado con un estado nuevo
        self._seen_action_states = set()
        self.expanded_count = 0

    # Métodos esperados por SimpleAI:
    def actions(self, state: int) -> List[int]:
        """
        Retorna lista de acciones desde 'state'. En este problema, cada acción es el id del nodo sucesor.
        Además, actualiza contador de nodos expandidos (aproximado).
        """
        if state not in self._seen_action_states:
            self._seen_action_states.add(state)
            self.expanded_count += 1

        # usar sucesors si es DiGraph, sino neighbors
        if hasattr(self.G, "successors") and isinstance(self.G, nx.DiGraph):
            neighbors = list(self.G.successors(state))
        else:
            neighbors = list(self.G.neighbors(state))
        # devolver vecinos (acciones)
        return neighbors

    def result(self, state: int, action: int) -> int:
        """
        Aplicar action (moverse al nodo 'action'). Devuelve nuevo estado (id del nodo).
        """
        return action

    def is_goal(self, state: int) -> bool:
        return state == self.goal

    def cost(self, state: int, action: int, new_state: int) -> float:
        """
        Costo de moverse de 'state' a 'new_state' (habitualmente equal a atributo 'length').
        Si hay múltiples aristas (MultiDiGraph), tomamos la mínima.
        """
        u = state
        v = new_state
        # edge data: puede haber varias si es MultiGraph
        default_weight = 1.0
        if self.G.has_edge(u, v):
            # networkx MultiDiGraph: G[u][v] es dict de keys -> attr
            data = self.G.get_edge_data(u, v)
            if data is None:
                return default_weight
            # if MultiGraph: choose minimum 'length' among parallel edges
            if isinstance(data, dict) and any("length" in ed for ed in data.values() if isinstance(ed, dict)):
                best = float("inf")
                for key, ed in data.items():
                    if isinstance(ed, dict):
                        l = ed.get("length", ed.get("travel_time", default_weight))
                        try:
                            l = float(l)
                        except Exception:
                            l = default_weight
                        if l < best:
                            best = l
                return best if best != float("inf") else default_weight
            else:
                # single edge
                ed = data
                if isinstance(ed, dict):
                    return float(ed.get("length", ed.get("travel_time", default_weight)))
                # fallback
                try:
                    return float(ed)
                except Exception:
                    return default_weight
        else:
            # no hay arista directa (esto no debería pasar en grafos válidos)
            return default_weight

    def heuristic(self, state: int) -> float:
        """
        Heurística admisible: distancia euclidiana en CRS proyectado (si disponible),
        sino haversine entre coordenadas lat/lon.
        """
        return node_distance_m(self.G_proj, state, self.goal)


def crear_problema(G: nx.Graph, origin: int, dest: int, proyected_graph: Optional[nx.Graph] = None) -> GraphSearchProblem:
    """
    Compatibilidad con código anterior: crea y devuelve un objeto
    `GraphSearchProblem` listo para pasarse a los algoritmos de SimpleAI.
    Si no se proporciona `proyected_graph` intenta reproyectar con OSMnx,
    y en caso de fallo usa el grafo original.
    """
    validate_node_in_graph(G, origin)
    validate_node_in_graph(G, dest)
    G_proj = proyected_graph
    if G_proj is None:
        try:
            import osmnx as ox

            G_proj = ox.project_graph(G)
        except Exception:
            G_proj = G
    return GraphSearchProblem(G, initial=origin, goal=dest, projected_graph=G_proj)


# ------------------------------
# Wrapper para ejecutar búsquedas y medir
# ------------------------------
def run_search(
    G: nx.Graph,
    source: int,
    target: int,
    algorithm: str = "astar",
    G_proj: Optional[nx.Graph] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Ejecuta uno de los algoritmos (simpleai) sobre el grafo G desde source a target.
    algorithm in {'bfs','dfs','ucs','iddfs','astar'}
    Retorna diccionario con keys:
      - algorithm, time_s, path (list nodes) or None, path_length_m, nodes_expanded, success (bool)
    """
    validate_node_in_graph(G, source)
    validate_node_in_graph(G, target)

    problem = GraphSearchProblem(G, initial=source, goal=target, projected_graph=G_proj or G)
    solver = algorithm.lower()

    start = time.perf_counter()
    result = None
    try:
        # Fast backend using networkx shortest-path implementations
        if globals().get('USE_FAST_BACKEND', True):
            if solver == "bfs":
                # shortest path by number of edges
                path = nx.shortest_path(G, source=source, target=target)
                class R:
                    def __init__(self, p):
                        self._p = p
                    def path(self):
                        return self._p
                result = R(path)
            elif solver == "dfs":
                # simple DFS until goal (not necessarily shortest) but faster than simpleai
                stack = [source]
                parent = {source: None}
                found = False
                while stack:
                    node = stack.pop()
                    if node == target:
                        found = True
                        break
                    for nb in G.successors(node) if hasattr(G, "successors") and isinstance(G, nx.DiGraph) else G.neighbors(node):
                        if nb not in parent:
                            parent[nb] = node
                            stack.append(nb)
                if found:
                    cur = target
                    path = []
                    while cur is not None:
                        path.append(cur)
                        cur = parent.get(cur)
                    path = list(reversed(path))
                    class R2:
                        def __init__(self, p):
                            self._p = p
                        def path(self):
                            return self._p
                    result = R2(path)
                else:
                    result = None
            elif solver == "ucs":
                # Dijkstra using 'travel_time' or 'length' if present
                weight = None
                for u, v, data in G.edges(data=True):
                    if data and data.get("travel_time") is not None:
                        weight = "travel_time"
                        break
                if weight is None:
                    weight = "length"
                path = nx.shortest_path(G, source=source, target=target, weight=weight)
                class R3:
                    def __init__(self, p):
                        self._p = p
                    def path(self):
                        return self._p
                result = R3(path)
            elif solver == "iddfs":
                # use local implementation
                result = iterative_deepening(problem)
            elif solver == "astar":
                # A* using networkx with a heuristic based on projected coords if available
                def heuristic(a, b):
                    try:
                        na = (G_proj or G).nodes[a]
                        nb = (G_proj or G).nodes[b]
                        if ("x" in na and "y" in na) and ("x" in nb and "y" in nb):
                            dx = na["x"] - nb["x"]
                            dy = na["y"] - nb["y"]
                            return (dx * dx + dy * dy) ** 0.5
                    except Exception:
                        pass
                    # fallback to haversine
                    acoord = (G.nodes[a].get("y", G.nodes[a].get("lat")), G.nodes[a].get("x", G.nodes[a].get("lon")))
                    bcoord = (G.nodes[b].get("y", G.nodes[b].get("lat")), G.nodes[b].get("x", G.nodes[b].get("lon")))
                    return haversine_meters(acoord, bcoord)

                path = nx.astar_path(G, source, target, heuristic=heuristic, weight="length")
                class R4:
                    def __init__(self, p):
                        self._p = p
                    def path(self):
                        return self._p
                result = R4(path)
        else:
            # fallback to original simpleai solvers
            if solver == "bfs":
                result = breadth_first(problem)
            elif solver == "dfs":
                result = depth_first(problem)
            elif solver == "ucs":
                result = uniform_cost(problem)
            elif solver == "iddfs":
                result = iterative_deepening(problem)
            elif solver == "astar":
                result = astar(problem)

        # si después de los intentos aún no hay resultado, el algoritmo fue desconocido
        if result is None:
            raise ValueError(f"Algoritmo desconocido o sin resultado: {algorithm}")
    except Exception as e:
        # capturamos fallas en la búsqueda
        end = time.perf_counter()
        return {
            "algorithm": algorithm,
            "time_s": end - start,
            "path": None,
            "path_length_m": None,
            "nodes_expanded": getattr(problem, "expanded_count", None),
            "success": False,
            "error": str(e),
        }
    end = time.perf_counter()
    total_time = end - start

    if result is None:
        return {
            "algorithm": algorithm,
            "time_s": total_time,
            "path": None,
            "path_length_m": None,
            "nodes_expanded": problem.expanded_count,
            "success": False,
        }

    # result.path() devuelve la lista de estados (nodos) desde inicial hasta objetivo
    try:
        path_states = result.path()
        # simpleai sometimes returns a list of states; ensure flat list of ints
        if len(path_states) > 0 and isinstance(path_states[0], tuple):
            # defensive: sometimes path contains (state, action) pairs in older versions
            path = [s if isinstance(s, int) else s[0] for s in path_states]
        else:
            path = [int(s) for s in path_states]
    except Exception:
        # fallback: intentar usar result.state y reconstruir (no ideal)
        try:
            path = [int(result.state)]
        except Exception:
            path = None

    path_length_m = None
    if path and len(path) >= 2:
        # sumar longitudes de aristas usando node_distance_m o atributos 'length' si existen
        total = 0.0
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            # preferir atributo 'length' si está
            if G.has_edge(u, v):
                data = G.get_edge_data(u, v)
                # data puede ser multi; obtener mínimo 'length'
                if isinstance(data, dict) and any(isinstance(ed, dict) for ed in data.values()):
                    best = float("inf")
                    for key, ed in data.items():
                        if isinstance(ed, dict):
                            l = ed.get("length", ed.get("travel_time", None))
                            if l is None:
                                continue
                            try:
                                l = float(l)
                            except Exception:
                                continue
                            if l < best:
                                best = l
                    if best != float("inf"):
                        total += best
                        continue
                # single edge case
                if isinstance(data, dict) and "length" in data:
                    try:
                        total += float(data.get("length"))
                        continue
                    except Exception:
                        pass
            # fallback: usar distancia entre nodos (proyectada o haversine)
            total += node_distance_m(G_proj or G, u, v)
        path_length_m = total

    return {
        "algorithm": algorithm,
        "time_s": total_time,
        "path": path,
        "path_length_m": path_length_m,
        "nodes_expanded": problem.expanded_count,
        "success": True if path else False,
    }


# ------------------------------
# Utils: generar parejas por distancia
# ------------------------------
def pick_pairs_by_distance(
    G: nx.Graph,
    G_proj: Optional[nx.Graph],
    n_pairs_each: int = 5,
    bands: List[Tuple[float, float]] = [(0, 1000), (1000, 5000), (5000, 9999999)],
    rng_seed: int = 0,
) -> Dict[str, List[Tuple[int, int, float]]]:
    """
    Selecciona parejas de nodos (u,v) cuya distancia se encuentre en cada banda (metros).
    Retorna dict: band_label -> list of tuples (u,v,dist_m)
    Estrategia:
      - muestrea aleatoriamente nodos y busca su nearest neighbor que cumpla banda.
    Nota: esto intenta obtener n_pairs_each por banda; si no encuentra, devuelve lo máximo hallado.
    """
    import random

    nodes = list(G.nodes)
    random.seed(rng_seed)
    res = {}
    for (dmin, dmax) in bands:
        found = []
        attempts = 0
        max_attempts = 10000
        while len(found) < n_pairs_each and attempts < max_attempts:
            a = random.choice(nodes)
            b = random.choice(nodes)
            if a == b:
                attempts += 1
                continue
            dist = node_distance_m(G_proj or G, a, b)
            if dist >= dmin and dist <= dmax:
                found.append((a, b, dist))
            attempts += 1
        key = f"{int(dmin)}_{int(dmax)}"
        res[key] = found
    return res


# ------------------------------
# Benchmark: ejecutar todos los algoritmos sobre parejas y guardar CSV
# ------------------------------
def benchmark_algorithms(
    G: nx.Graph,
    G_proj: Optional[nx.Graph],
    out_csv: Optional[str] = None,
    n_pairs_each: int = 5,
    algorithms: Optional[List[str]] = None,
):
    """
    Ejecuta BFS/DFS/UCS/IDDFS/A* sobre pares generados y guarda CSV con resultados.
    """
    if out_csv is None:
        out_csv = os.path.join(DATA_DIR, "routing_benchmark.csv")
    if algorithms is None:
        algorithms = ["bfs", "dfs", "ucs", "iddfs", "astar"]

    pairs = pick_pairs_by_distance(G, G_proj, n_pairs_each=n_pairs_each)
    rows = []
    test_id = 0
    for band_label, pairlist in pairs.items():
        for (u, v, dist) in pairlist:
            test_id += 1
            for algo in algorithms:
                print(f"[benchmark] test {test_id} band={band_label} pair=({u},{v}) algo={algo} ...")
                out = run_search(G, u, v, algorithm=algo, G_proj=G_proj)
                row = {
                    "test_id": test_id,
                    "band": band_label,
                    "source": int(u),
                    "target": int(v),
                    "pair_dist_m": float(dist),
                    "algorithm": algo,
                    "time_s": float(out.get("time_s", None) or 0.0),
                    "success": bool(out.get("success", False)),
                    "path_length_m": float(out.get("path_length_m", 0.0) or 0.0),
                    "nodes_expanded": int(out.get("nodes_expanded", 0) or 0),
                }
                rows.append(row)
                print(
                    f"    -> time {format_seconds(row['time_s'])} | success={row['success']} | path_len={row['path_length_m']:.1f} m | nodes_exp={row['nodes_expanded']}"
                )
    save_csv(rows, out_csv)
    print(f"[benchmark] resultados guardados en: {out_csv}")
    return rows


# ------------------------------
# Compatibilidad: select_node_pairs_by_distance + evaluate_algorithms_on_pairs
# ------------------------------
def select_node_pairs_by_distance(G: nx.Graph, n_pairs_per_bucket: int = 5, seed: int = 1):
    """
    Compat wrapper para la antigua API usada por testRutas.py.
    Devuelve buckets con keys 'short','medium','long' cada una conteniendo
    listas de tuplas (u,v).
    """
    # bands: short <1000, medium 1000-5000, long >5000
    G_proj = None
    try:
        import osmnx as ox
        G_proj = ox.project_graph(G)
    except Exception:
        G_proj = G

    raw = pick_pairs_by_distance(G, G_proj, n_pairs_each=n_pairs_per_bucket, rng_seed=seed)
    # raw keys are like '0_1000', '1000_5000', '5000_9999999'
    buckets = {"short": [], "medium": [], "long": []}
    for k, lst in raw.items():
        # determine which bucket
        parts = k.split("_")
        try:
            dmin = int(parts[0])
            dmax = int(parts[1])
        except Exception:
            continue
        if dmax <= 1000:
            label = "short"
        elif dmax <= 5000:
            label = "medium"
        else:
            label = "long"
        # raw entries are (u,v,dist)
        buckets[label] = [(u, v) for (u, v, _) in lst]
    return buckets


def evaluate_algorithms_on_pairs(G: nx.Graph, pairs: List[Tuple[int, int]], out_csv: Optional[str] = None):
    """
    Ejecuta BFS/DFS/UCS/IDDFS/A* sobre cada par en `pairs`.
    `pairs` debe ser una lista de tuplas (start, goal).
    Guarda CSV si `out_csv` se proporciona.
    Retorna lista de resultados (diccionarios) compatibles con testRutas.py expectations.
    """
    runners = ["bfs", "dfs", "ucs", "iddfs", "astar"]
    results = []
    for (a, b) in pairs:
        for algo in runners:
            out = run_search(G, a, b, algorithm=algo)
            row = {
                "start": int(a),
                "goal": int(b),
                "algorithm": out.get("algorithm", algo),
                "success": bool(out.get("success", False)),
                "time_s": float(out.get("time_s", 0.0) or 0.0),
                "nodes_expanded": int(out.get("nodes_expanded", 0) or 0),
                "route_length_m": float(out.get("path_length_m", 0.0) or 0.0),
                "route_time_s": float(out.get("path_time_s", 0.0) or 0.0) if out.get("path_time_s", None) is not None else None,
                "error": out.get("error", None),
            }
            results.append(row)

    if out_csv:
        ensure_dir(os.path.dirname(out_csv) or ".")
        save_csv(results, out_csv)

    return results


# ------------------------------
# CLI / Auto-test
# ------------------------------
def demo_with_graph(G: nx.Graph, G_proj: Optional[nx.Graph] = None):
    """
    Demo rápido: selecciona 3 pares (cercano, medio, lejos) y corre cada algoritmo.
    """
    print("[demo] generando parejas de prueba...")
    pairs = pick_pairs_by_distance(G, G_proj, n_pairs_each=3)
    # elegimos primera pareja disponible de cada banda
    chosen = []
    for k, v in pairs.items():
        if v:
            chosen.append(v[0])
    print(f"[demo] parejas seleccionadas: {chosen}")
    for (u, v, dist) in chosen:
        print(f"\n[demo] pair {u} -> {v} dist={dist:.1f} m")
        for algo in ["bfs", "dfs", "ucs", "iddfs", "astar"]:
            res = run_search(G, u, v, algorithm=algo, G_proj=G_proj)
            print(
                f"  {algo:6s} time={format_seconds(res['time_s'])} success={res['success']} nodes_exp={res['nodes_expanded']} path_len_m={res['path_length_m']}"
            )


def _parse_args():
    parser = argparse.ArgumentParser(description="Demo / benchmark de algoritmos de búsqueda (SimpleAI).")
    parser.add_argument("--graphml", help="Ruta a graphml guardado (si se quiere cargar en lugar de descargar).")
    parser.add_argument("--place", help="Dirección / lugar para descargar grafo si no se usa graphml.", default=None)
    parser.add_argument("--dist", help="Radio en metros para descargar (si se usa place).", type=int, default=5000)
    parser.add_argument("--mode", choices=["demo", "benchmark"], default="demo", help="Ejecutar demo rápido o benchmark completo.")
    parser.add_argument("--out", help="Ruta CSV salida (benchmark).", default=None)
    parser.add_argument("--n", help="Número de parejas por banda (benchmark).", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    # comportamiento simple para ejecución directa
    args = _parse_args()
    if args.graphml:
        print(f"[main] cargando graphml desde {args.graphml} ...")
        G = nx.read_graphml(args.graphml)
        G = nx.convert_node_labels_to_integers(G, label_attribute="osmid") if not any(isinstance(n, int) for n in G.nodes) else G
        # Nota: si el graphml viene con atributos 'x','y' ya estará proyectado
        G_proj = G
    else:
        if not args.place:
            raise SystemExit("Debes proporcionar --place o --graphml para demos/benchmarks.")
        import osmnx as ox
        print(f"[main] descargando grafo para {args.place} (dist={args.dist}) ...")
        G = ox.graph_from_address(args.place, dist=args.dist, network_type="drive")
        G_proj = ox.project_graph(G)

    if args.mode == "demo":
        demo_with_graph(G, G_proj)
    else:
        benchmark_algorithms(G, G_proj, out_csv=args.out, n_pairs_each=args.n)

