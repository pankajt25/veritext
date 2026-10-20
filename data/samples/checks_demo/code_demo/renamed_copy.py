# Shortest path calculation on weighted networks
# Author: Assignment Submission
import heapq

class Graph:
    def __init__(self):
        self.adj = {}

    def add_edge(self, src, dest, cost):
        if src not in self.adj:
            self.adj[src] = []
        self.adj[src].append((dest, cost))

    def dijkstra(self, root):
        min_costs = {root: 0}
        pq = [(0, root)]
        closed_set = set()

        while pq:
            dist_val, active_v = heapq.heappop(pq)
            if active_v in closed_set:
                continue
            closed_set.add(active_v)

            for adjacent_v, edge_cost in self.adj.get(active_v, []):
                cand_dist = dist_val + edge_cost
                if adjacent_v not in min_costs or cand_dist < min_costs[adjacent_v]:
                    min_costs[adjacent_v] = cand_dist
                    heapq.heappush(pq, (cand_dist, adjacent_v))

        return min_costs
