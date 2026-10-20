# Graph pathfinding implementation using Dijkstra's algorithm
import heapq

class Graph:
    def __init__(self):
        self.adj = {}

    def add_edge(self, u, v, weight):
        if u not in self.adj:
            self.adj[u] = []
        self.adj[u].append((v, weight))

    def dijkstra(self, start_node):
        distances = {start_node: 0}
        priority_queue = [(0, start_node)]
        visited = set()

        while priority_queue:
            curr_dist, curr_node = heapq.heappop(priority_queue)
            if curr_node in visited:
                continue
            visited.add(curr_node)

            for neighbor, weight in self.adj.get(curr_node, []):
                distance = curr_dist + weight
                if neighbor not in distances or distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(priority_queue, (distance, neighbor))

        return distances
