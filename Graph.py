from Node import *
from Edge import *
import sys

class Graph:
	def __init__(self):
		# list of nodes and edges in the graph
		self.nodes = []
		self.edges = []

	# draw the nodes and edges that connect all of the open cells on top of the blocks to show the node graph
	def draw_graph(self):
		# draw the nodes
		for node in self.nodes:
			node.draw_node()

		# draw the edges
		for edge in self.edges:
			edge.draw_edge()

	def add_node(self, node):
		self.nodes.append(node)

	def add_edge(self, edge):
		self.edges.append(edge)

	def get_nodes(self):
		return self.nodes

	def get_edges(self):
		return self.edges

	def set_nodes(self, nodes):
		self.nodes = nodes

	def set_edges(self, edges):
		self.edges = edges

	def reset_nodes(self):
		for node in self.nodes:
			node.set_visited(False)
			node.set_removed_from_path(False)
			node.set_being_looked_at(False)
			node.set_in_queue(False)
			node.set_cost(sys.maxsize)