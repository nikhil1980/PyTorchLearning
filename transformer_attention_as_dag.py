"""
 This code implements basic attention operation on which present days Transformers are based

  Attention(Q, K, V) = SOFTMAX ((Q.K^T)/SQRT(rank(Q)).V

  We basically create a DAG with arbit nodes and edges.
  Each node will have three things:
        -  A query Q (what task we ask this node to do)
        -  A key K (what info this node has)
        -  A value V (what response this node sends against Q)

  Attention is basically updating our response D based on value V.
  This V is calculated selectively as per how each key K is important
  for a given query Q
"""
import numpy as np
from math import sqrt


class Node:
    def __init__(self, dim):
        # the basic data stored
        self.dim = dim
        self.data = np.random.randn(self.dim)

        # Weights of Q, K, V (dim X dim SQ MAT)
        self.w_query = np.random.randn(self.dim, self.dim)
        self.w_key = np.random.randn(self.dim, self.dim)
        self.w_value = np.random.randn(self.dim, self.dim)

    def query(self):
        # What am I asked?
        return self.w_query @ self.data

    def key(self):
        # What info I have
        return self.w_key @ self.data

    def value(self):
        # What is my updated response based on Q and K
        return self.w_value @ self.data


class Graph:
    def __init__(self, num_nodes, num_edges, dim):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.dim = dim

        # Create Graph
        self.nodes = [Node(dim) for _ in range(self.num_nodes)]
        rand_edges = lambda: np.random.randint(len(self.nodes))
        self.edges = [[rand_edges(), rand_edges()] for _ in range(self.num_edges)]

    def get_data(self):
        for i, n in enumerate(self.nodes):
            print(f"Node: {i + 1} has data {n.data}")

    def run(self):
        attn_list = list()
        for i, n in enumerate(self.nodes):
            # What is it that you are looking for
            q = n.query()

            # Find all edges that is connected to this node. Rt now it is
            # random but in practice it depends upon your problem.
            inputs = [self.nodes[src] for (src, dest) in self.edges if dest == i]
            if len(inputs) == 0:
                continue  # Ignore nodes with no edges

            # Find keys. What context do I have for your query?
            keys = [m.key() for m in inputs]

            # Find score by DOT product
            scores = [k.dot(q) / sqrt(self.dim) for k in keys]

            # Normalize them via SOFTMAX
            scores = np.exp(scores)
            scores = scores / np.sum(scores)

            # Based on key and query, I update my response on what I can offer
            values = [m.value() for m in inputs]

            # Calculated weighted sum or ATTENTION
            attn = sum(s * v for s, v in zip(scores, values))
            attn_list.append(attn)

        for n, a in zip(self.nodes, attn_list):
            # Residual Connection
            # out = in + F(in)
            n.data = n.data + a


def main():
    my_graph = Graph(20, 40, 20)
    my_graph.run()
    my_graph.get_data()


if __name__ == '__main__':
    main()
