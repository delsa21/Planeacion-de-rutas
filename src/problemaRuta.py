# src/problemaRuta.py
"""
Problema de búsqueda sobre un grafo de OSMnx usando SimpleAI.

Proporciona:
 - Clase ProblemaRuta(SearchProblem)
 - Función crear_problema(G, start, goal)
"""

from simpleai.search import SearchProblem
from src.utils import haversine_meters, validate_node_in_graph


class ProblemaRuta(SearchProblem):
    """
    Define un problema de búsqueda estándar para SimpleAI.
    Usa el grafo de OSMnx:
        - actions(): nodos sucesores
        - result(): transición al siguiente nodo
        - cost(): longitud real de la arista (metros)
        - heuristic(): distancia geodésica hacia la meta (metros)
    """

    def __init__(self, G, start, goal):
        self.G = G
        self.start = start
        self.goal = goal

        validate_node_in_graph(G, start)
        validate_node_in_graph(G, goal)

        super().__init__(initial=start)

    # ----------------------------------------------------
    def actions(self, state):
        """Regresa los nodos sucesores (IDs) del grafo."""
        return list(self.G.successors(state))

    # ----------------------------------------------------
    def result(self, state, action):
        """El resultado de tomar la acción es el nodo destino."""
        return action

    # ----------------------------------------------------
    def is_goal(self, state):
        """Determina si el nodo actual es la meta."""
        return state == self.goal

    # ----------------------------------------------------
    def cost(self, state1, action, state2):
        """
        Costo = longitud real de la arista (metros).
        OSMnx almacena multiaristas, tomamos la primera.
        """
        data = self.G.get_edge_data(state1, state2)

        if data is None:
            # No debería ocurrir si successors() es correcto
            return float("inf")

        # data es un diccionario: {0:{...}, 1:{...}} en multigraph
        edge_info = next(iter(data.values()))

        length = edge_info.get("length", None)
        if length is not None:
            return length

        # Si no tiene 'length', usamos distancia geodésica
        lat1, lon1 = self.G.nodes[state1]["y"], self.G.nodes[state1]["x"]
        lat2, lon2 = self.G.nodes[state2]["y"], self.G.nodes[state2]["x"]
        return haversine_meters((lat1, lon1), (lat2, lon2))

    # ----------------------------------------------------
    def heuristic(self, state):
        """
        Heurística para A*:
        distancia geodésica (metros) al nodo objetivo.
        Es admisible y consistente.
        """
        lat1 = self.G.nodes[state]["y"]
        lon1 = self.G.nodes[state]["x"]

        lat2 = self.G.nodes[self.goal]["y"]
        lon2 = self.G.nodes[self.goal]["x"]

        return haversine_meters((lat1, lon1), (lat2, lon2))


# ==========================================================
# Función auxiliar
# ==========================================================
def crear_problema(G, start, goal):
    """
    Verifica nodos e instancia el problema.
    """
    return ProblemaRuta(G, start, goal)


# ==========================================================
# Auto-test (solo si ejecutas este archivo directamente)
# ==========================================================
if __name__ == "__main__":
    print("Este archivo se usa como módulo desde testRutas.py o main.py.")
    print("No ejecuta rutas por sí solo.")
