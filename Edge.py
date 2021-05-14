import pygame
import math
from Node import *
from Colors import *

class Edge:
	def __init__(self, start_node, end_node, WIN):
		# the starting node and end nodes in the edge, their centers are used to draw a line to connect them as an edge
		self.start_node = start_node
		self.end_node = end_node

		# window to draw to the screen
		self.WIN = WIN

		# the edge length from the start_node to the end_node, this is the difference in rows or columns
		self.length = self.find_length()

	def draw_edge(self):
		# if both nodes have been visited then draw the edge line as blue, otherwise draw it red
		if self.start_node.visited and self.end_node.visited:
			color = BLUE
		else:
			color = RED

		# draw the line from the center of the start node to the center of the end node in the appropriate color
		pygame.draw.line(self.WIN, color, self.start_node.get_center(), self.end_node.get_center())		

		# draw the center point of the start node
		pygame.draw.circle(self.WIN, color, self.start_node.center, radius=3, width=0)

		# draw the center point of the end node
		pygame.draw.circle(self.WIN, color, self.end_node.center, radius=3, width=0)


	def find_length(self):
		# difference between either rows or columns between the nodes
		length = 0
		# if the start node and the end node are on the same row, the difference between their columns is their distance
		if self.start_node.row == self.end_node.row:
			length = math.fabs(self.end_node.col - self.start_node.col)

		# if start node and end node are on the same column, the difference between their rows is their distance
		if self.start_node.col == self.end_node.col:
			length = math.fabs(self.end_node.row - self.start_node.row)

		# could also have gotten the column difference and added it with the row difference since one of them is always going to be 0
		# return the edge length
		return int(length)

	def get_start_node(self):
		return self.start_node

	def get_end_node(self):
		return self.end_node

	def get_length(self):
		return self.length

	def set_start_node(self, start_node):
		self.start_node = start_node

	def set_end_node(self, end_node):
		self.end_node = end_node

	def set_length(self, length):
		self.length = length

	def __repr__(self):
		s = f"\nstarting node: ({self.start_node.get_row()},{self.start_node.get_col()}), cost: {self.start_node.get_cost()}, visited: {self.start_node.get_visited()}"
		s += f"\nending node: ({self.end_node.get_row()},{self.end_node.get_col()}), cost: {self.end_node.get_cost()}, visited: {self.end_node.get_visited()}"
		s += f"\nedge length: {self.length}"
		return s