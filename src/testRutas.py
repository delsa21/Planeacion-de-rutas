import time
import random
import math
from simpleai.search import (
    breadth_first,
    depth_first,
    uniform_cost,
    astar,
    iterative_limited_depth_first
)
from src.problemaRuta import crear_problema
from src.utils import save_csv, format_seconds, ensure_dir

def calcular_distancia_aerea(G, n1, n2):
    """Calcula distancia línea recta aproximada entre dos nodos"""
    x1, y1 = G.nodes[n1]['x'], G.nodes[n1]['y']
    x2, y2 = G.nodes[n2]['x'], G.nodes[n2]['y']
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def run_routing_tests(G, num_pairs=3, out_csv="datos/rutas_tests.csv"):

    print("\n--- Iniciando Comparativa Rápida ---")
    nodes = list(G.nodes())
    results = []

    # Lista de algoritmos a probar
    algorithms = [
        ("BFS", breadth_first),
        ("DFS", depth_first),
        ("UCS", uniform_cost),
        ("A*", astar),
        ("IDDFS", iterative_limited_depth_first) 
    ]

    count = 0
    max_intentos = num_pairs * 50
    intentos = 0
    
    while count < num_pairs and intentos < max_intentos:
        intentos += 1
        origin = random.choice(nodes)
        dest = random.choice(nodes)
    
        try:
            dist_aprox = calcular_distancia_aerea(G, origin, dest)
        except:
            continue
        
        if origin == dest or dist_aprox > 600: 
            continue

        count += 1
        print(f"\n[Prueba {count}/{num_pairs}] Distancia aprox: {dist_aprox:.1f}m")
        
        problem = crear_problema(G, origin, dest)

        for name, algo_func in algorithms:
            print(f"  > {name}...", end=" ", flush=True)
            
            start_t = time.time()
            try:
                # graph_search=True es VITAL para evitar ciclos infinitos
                res = algo_func(problem, graph_search=True)
                
                elapsed = time.time() - start_t
                costo = res.cost if res else 0
                path_len = len(res.path()) if res else 0
                
                print(f"OK ({elapsed:.2f}s)")
                
                results.append({
                    "test_id": count,
                    "origen": origin,
                    "destino": dest,
                    "algoritmo": name,
                    "tiempo_seg": elapsed,
                    "costo_metros": costo,
                    "nodos_visitados": path_len
                })

            except Exception as e:
                print(f"X ({e})")
                results.append({"test_id": count, "algoritmo": name, "tiempo_seg": -1})

    ensure_dir("datos")
    save_csv(results, out_csv)
    print(f"\nResultados guardados en {out_csv}")