import pygame
import sys
from Colors import *
class Node:
	def __init__(self, row, col, cell_size, WIN):
		# what row and column the node is at
		self.row = row
		self.col = col

		# the radius of the circle for the node and where the center of that circle is in x,y
		self.radius = cell_size // 2
		self.center = (((self.col * cell_size) + self.radius, (self.row * cell_size) + self.radius))
		
		# window to draw the node onto
		self.WIN = WIN

		# if the node has been visited yet or not
		self.visited = False

		# if the node was in the path and later remove from it once a dead-end was found, used for depth first search
		self.removed_from_path = False

		# if the current node is being looked at, only used for animation in breadth first search
		self.being_looked_at = False

		# if the node is in the breadth first search queue to eventually be looked at
		self.in_queue = False

		# distance from start node in terms of pixels needed to be traveled, not the number of nodes from start node, start at essentially infinity
		self.cost = sys.maxsize

	def draw_node(self, color=None):
		# get the color depending on if the node is being looked at, in the queue, has been removed from the path, or visited, or not
		if self.being_looked_at:
			color = RED
		elif self.in_queue:
			color = YELLOW
		elif self.removed_from_path:
			color = GRAY
		elif self.visited:
			color = BLUE
		else:
			color = GREEN

		# draw a crcle on screen WIN with color at center with a radius and fill it in
		pygame.draw.circle(self.WIN, color, self.center, self.radius, width=4)

	def get_row(self):
		return self.row

	def get_col(self):
		return self.col

	def get_center(self):
		return self.center

	def get_visited(self):
		return self.visited

	def get_removed_from_path(self):
		return self.removed_from_path

	def get_being_looked_at(self):
		return self.being_looked_at

	def get_in_queue(self):
		return self.in_queue

	def get_cost(self):
		return self.cost

	def set_row(self, row):
		self.row = row

	def set_col(self, col):
		self.col = col

	def set_center(self, center):
		self.center= center

	def set_visited(self, visited):
		self.visited = visited

	def set_removed_from_path(self, removed_from_path):
		self.removed_from_path = removed_from_path

	def set_being_looked_at(self, being_looked_at):
		self.being_looked_at = being_looked_at

	def set_in_queue(self, in_queue):
		self.in_queue = in_queue

	def set_cost(self, cost):
		self.cost = cost

	def visit(self):
		self.set_visited(True)

	def remove(self):
		self.set_removed_from_path(True)

	def get_num_nodes():
		return Node.num_nodes

	def __repr__(self):
		return f"pos: ({self.row},{self.col}), cost: {self.cost}, visited: {self.visited}, removed: {self.removed_from_path}, being looked at: {self.being_looked_at}, in queue: {self.in_queue}"