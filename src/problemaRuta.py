# src/problemaRuta.py
from __future__ import annotations
import time
import os
import argparse
from typing import Any, List, Tuple, Optional, Dict
import networkx as nx
from simpleai.search import (
    SearchProblem,
    breadth_first,
    depth_first,
    uniform_cost,
    astar,
)
from src.utils import (
    haversine_meters,
    timeit,
    ensure_dir,
    save_csv,
    format_seconds,
    validate_node_in_graph,
)

# planeador de rutas usando simpleai (bfs, dfs, ucs, iddfs, a*).


DATA_DIR = "datos"
ensure_dir(DATA_DIR)


# dibujo de ruta sobre mapa
def plot_route_on_map(G, path, out_path: Optional[str] = None, show: bool = False):
    # dibuja ruta y opcionalmente la guarda en archivo png
    try:
        import osmnx as ox
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[Plot_route_on_map] Error importando librerías de visualización: {e}")
        return False

    if not path:
        print("[Plot_route_on_map] Ruta vacía, nada para graficar")
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
        print(f"[Plot_route_on_map] Error al graficar ruta: {repr(e)}")
        return False

    if out_path:
        ensure_dir(os.path.dirname(out_path) or ".")
        try:
            fig.savefig(out_path, dpi=200, bbox_inches="tight")
            print(f"[Plot_route_on_map] Imagen guardada en: {out_path}")
        except Exception as e:
            print(f"[Plot_route_on_map] No se pudo guardar la imagen: {e}")

    if show:
        try:
            plt.show()
        except Exception as e:
            print(f"[Plot_route_on_map] Error mostrando figura: {e}")

    try:
        plt.close(fig)
    except Exception:
        pass

    return True


# distancia entre nodos
def node_distance_m(G_proj: nx.Graph, u: int, v: int) -> float:
    # distancia aproximada entre nodos usando x,y si existen; si no, haversine
    du = G_proj.nodes[u]
    dv = G_proj.nodes[v]

    if ("x" in du and "y" in du) and ("x" in dv and "y" in dv):
        dx = du["x"] - dv["x"]
        dy = du["y"] - dv["y"]
        return (dx * dx + dy * dy) ** 0.5

    a = (du.get("y", du.get("lat")), du.get("x", du.get("lon")))
    b = (dv.get("y", dv.get("lat")), dv.get("x", dv.get("lon")))
    return haversine_meters(a, b)


# iddfs local
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


# problema simpleai para grafos
class GraphSearchProblem(SearchProblem):

    # simpleai searchproblem usando nodos de un grafo networkx

    def __init__(
        self,
        G: nx.Graph,
        initial: int,
        goal: int,
        projected_graph: Optional[nx.Graph] = None,
    ):
        super(GraphSearchProblem, self).__init__(initial_state=initial)
        self.G = G
        self.initial = initial
        self.goal = goal
        self.G_proj = projected_graph or G
        self._seen_action_states = set()
        self.expanded_count = 0

    def actions(self, state: int) -> List[int]:
        if state not in self._seen_action_states:
            self._seen_action_states.add(state)
            self.expanded_count += 1

        if hasattr(self.G, "successors") and isinstance(self.G, nx.DiGraph):
            return list(self.G.successors(state))
        return list(self.G.neighbors(state))

    def result(self, state: int, action: int) -> int:
        return action

    def is_goal(self, state: int) -> bool:
        return state == self.goal

    def cost(self, state: int, action: int, new_state: int) -> float:
        u = state
        v = new_state
        default_weight = 1.0

        if self.G.has_edge(u, v):
            data = self.G.get_edge_data(u, v)
            if data is None:
                return default_weight

            if isinstance(data, dict) and any(
                "length" in ed for ed in data.values() if isinstance(ed, dict)
            ):
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

            if isinstance(data, dict):
                return float(
                    data.get("length", data.get("travel_time", default_weight))
                )

            try:
                return float(data)
            except Exception:
                return default_weight

        return default_weight

    def heuristic(self, state: int) -> float:
        return node_distance_m(self.G_proj, state, self.goal)


# crear problema
def crear_problema(
    G: nx.Graph, origin: int, dest: int, proyected_graph: Optional[nx.Graph] = None
) -> GraphSearchProblem:
    # crea un objeto graphsearchproblem para simpleai
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


# ejecutar búsquedas
def run_search(
    G: nx.Graph,
    source: int,
    target: int,
    algorithm: str = "astar",
    G_proj: Optional[nx.Graph] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:

    validate_node_in_graph(G, source)
    validate_node_in_graph(G, target)

    problem = GraphSearchProblem(
        G, initial=source, goal=target, projected_graph=G_proj or G
    )
    solver = algorithm.lower()

    start = time.perf_counter()
    result = None

    try:
        if globals().get("USE_FAST_BACKEND", True):

            if solver == "bfs":
                path = nx.shortest_path(G, source=source, target=target)

                class R:
                    def __init__(self, p):
                        self._p = p

                    def path(self):
                        return self._p

                result = R(path)

            elif solver == "dfs":
                stack = [source]
                parent = {source: None}
                found = False

                while stack:
                    node = stack.pop()
                    if node == target:
                        found = True
                        break
                    for nb in (
                        G.successors(node)
                        if hasattr(G, "successors")
                        else G.neighbors(node)
                    ):
                        if nb not in parent:
                            parent[nb] = node
                            stack.append(nb)

                if found:
                    cur = target
                    path = []
                    while cur is not None:
                        path.append(cur)
                        cur = parent.get(cur)
                    path.reverse()

                    class R2:
                        def __init__(self, p):
                            self._p = p

                        def path(self):
                            return self._p

                    result = R2(path)
                else:
                    result = None

            elif solver == "ucs":
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
                result = iterative_deepening(problem)

            elif solver == "astar":

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

                    acoord = (
                        G.nodes[a].get("y", G.nodes[a].get("lat")),
                        G.nodes[a].get("x", G.nodes[a].get("lon")),
                    )
                    bcoord = (
                        G.nodes[b].get("y", G.nodes[b].get("lat")),
                        G.nodes[b].get("x", G.nodes[b].get("lon")),
                    )
                    return haversine_meters(acoord, bcoord)

                path = nx.astar_path(
                    G, source, target, heuristic=heuristic, weight="length"
                )

                class R4:
                    def __init__(self, p):
                        self._p = p

                    def path(self):
                        return self._p

                result = R4(path)

        else:
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

        if result is None:
            raise ValueError(f"Algoritmo desconocido o sin resultado: {algorithm}")

    except Exception as e:
        end = time.perf_counter()
        return {
            "algorithm": algorithm,
            "time_s": end - start,
            "path": None,
            "path_length_m": None,
            "nodes_expanded": problem.expanded_count,
            "success": False,
            "error": str(e),
        }

    end = time.perf_counter()
    total_time = end - start

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

    path_length_m = None
    if path and len(path) >= 2:
        total = 0.0

        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]

            if G.has_edge(u, v):
                data = G.get_edge_data(u, v)

                if isinstance(data, dict) and any(
                    isinstance(ed, dict) for ed in data.values()
                ):
                    best = float("inf")
                    for _, ed in data.items():
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

                if isinstance(data, dict) and "length" in data:
                    try:
                        total += float(data.get("length"))
                        continue
                    except Exception:
                        pass

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


# generar pares por distancia
def pick_pairs_by_distance(
    G: nx.Graph,
    G_proj: Optional[nx.Graph],
    n_pairs_each: int = 5,
    bands: List[Tuple[float, float]] = [(0, 1000), (1000, 5000), (5000, 9999999)],
    rng_seed: int = 0,
) -> Dict[str, List[Tuple[int, int, float]]]:
    # selecciona pares (u,v) cuya distancia cae en cada banda definida
    import random

    nodes = list(G.nodes)
    random.seed(rng_seed)
    res = {}

    for dmin, dmax in bands:
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
            if dmin <= dist <= dmax:
                found.append((a, b, dist))

            attempts += 1

        key = f"{int(dmin)}_{int(dmax)}"
        res[key] = found

    return res


# ejecutar benchmark
def benchmark_algorithms(
    G: nx.Graph,
    G_proj: Optional[nx.Graph],
    out_csv: Optional[str] = None,
    n_pairs_each: int = 5,
    algorithms: Optional[List[str]] = None,
):
    # corre bfs, dfs, ucs, iddfs, a* sobre varias parejas

    if out_csv is None:
        out_csv = os.path.join(DATA_DIR, "routing_benchmark.csv")

    if algorithms is None:
        algorithms = ["bfs", "dfs", "ucs", "iddfs", "astar"]

    pairs = pick_pairs_by_distance(G, G_proj, n_pairs_each=n_pairs_each)
    rows = []
    test_id = 0

    for band_label, pairlist in pairs.items():
        for u, v, dist in pairlist:
            test_id += 1

            for algo in algorithms:
                print(
                    f"[Benchmark] Test {test_id} banda={band_label} nodos=({u},{v}) algoritmo={algo}"
                )

                out = run_search(G, u, v, algorithm=algo, G_proj=G_proj)

                row = {
                    "test_id": test_id,
                    "band": band_label,
                    "source": int(u),
                    "target": int(v),
                    "pair_dist_m": float(dist),
                    "algorithm": algo,
                    "time_s": float(out.get("time_s", 0.0)),
                    "success": bool(out.get("success", False)),
                    "path_length_m": float(out.get("path_length_m", 0.0)),
                    "nodes_expanded": int(out.get("nodes_expanded", 0)),
                }

                rows.append(row)

                print(
                    f"    -> Tiempo {format_seconds(row['time_s'])} | éxito={row['success']} | ruta={row['path_length_m']:.1f} m | expandidos={row['nodes_expanded']}"
                )

    save_csv(rows, out_csv)
    print(f"[Benchmark] Resultados guardados en {out_csv}")
    return rows


# compatibilidad
def select_node_pairs_by_distance(
    G: nx.Graph, n_pairs_per_bucket: int = 5, seed: int = 1
):
    # compatibilidad con testRutas antiguo
    try:
        import osmnx as ox

        G_proj = ox.project_graph(G)
    except Exception:
        G_proj = G

    raw = pick_pairs_by_distance(
        G, G_proj, n_pairs_each=n_pairs_per_bucket, rng_seed=seed
    )
    buckets = {"short": [], "medium": [], "long": []}

    for k, lst in raw.items():
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

        buckets[label] = [(u, v) for (u, v, _) in lst]

    return buckets


def evaluate_algorithms_on_pairs(
    G: nx.Graph, pairs: List[Tuple[int, int]], out_csv: Optional[str] = None
):
    # ejecuta bfs, dfs, ucs, iddfs, a* sobre pares definidos
    runners = ["bfs", "dfs", "ucs", "iddfs", "astar"]
    results = []

    for a, b in pairs:
        for algo in runners:
            out = run_search(G, a, b, algorithm=algo)

            row = {
                "start": int(a),
                "goal": int(b),
                "algorithm": out.get("algorithm", algo),
                "success": bool(out.get("success", False)),
                "time_s": float(out.get("time_s", 0.0)),
                "nodes_expanded": int(out.get("nodes_expanded", 0)),
                "route_length_m": float(out.get("path_length_m", 0.0)),
                "route_time_s": (
                    float(out.get("path_time_s", 0.0))
                    if out.get("path_time_s", None) is not None
                    else None
                ),
                "error": out.get("error", None),
            }

            results.append(row)

    if out_csv:
        ensure_dir(os.path.dirname(out_csv) or ".")
        save_csv(results, out_csv)

    return results


# demo
def demo_with_graph(G: nx.Graph, G_proj: Optional[nx.Graph] = None):
    # demo rápido sobre 3 parejas: corta, media, larga
    print("[Demo] Generando parejas de prueba")
    pairs = pick_pairs_by_distance(G, G_proj, n_pairs_each=3)

    chosen = []
    for _, v in pairs.items():
        if v:
            chosen.append(v[0])

    print(f"[Demo] Parejas seleccionadas: {chosen}")

    for u, v, dist in chosen:
        print(f"\n[Demo] Par {u} -> {v} dist={dist:.1f} m")

        for algo in ["bfs", "dfs", "ucs", "iddfs", "astar"]:
            res = run_search(G, u, v, algorithm=algo, G_proj=G_proj)
            print(
                f"  {algo:6s} tiempo={format_seconds(res['time_s'])} éxito={res['success']} expandidos={res['nodes_expanded']} ruta_m={res['path_length_m']}"
            )


# cli
def _parse_args():
    parser = argparse.ArgumentParser(
        description="demo / benchmark de algoritmos de búsqueda"
    )
    parser.add_argument("--graphml", help="ruta de archivo graphml a cargar")
    parser.add_argument(
        "--place", help="lugar o dirección para descargar grafo", default=None
    )
    parser.add_argument("--dist", help="radio de descarga", type=int, default=5000)
    parser.add_argument("--mode", choices=["demo", "benchmark"], default="demo")
    parser.add_argument("--out", help="archivo csv salida", default=None)
    parser.add_argument("--n", help="número de parejas por banda", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.graphml:
        print(f"[Main] Cargando graphml desde {args.graphml}")
        G = nx.read_graphml(args.graphml)
        G = (
            nx.convert_node_labels_to_integers(G, label_attribute="osmid")
            if not any(isinstance(n, int) for n in G.nodes)
            else G
        )
        G_proj = G
    else:
        if not args.place:
            raise SystemExit("Debes proporcionar --place o --graphml")

        import osmnx as ox

        print(f"[Main] Descargando grafo para {args.place} (dist={args.dist})")
        G = ox.graph_from_address(args.place, dist=args.dist, network_type="drive")
        G_proj = ox.project_graph(G)

    if args.mode == "demo":
        demo_with_graph(G, G_proj)
    else:
        benchmark_algorithms(G, G_proj, out_csv=args.out, n_pairs_each=args.n)
