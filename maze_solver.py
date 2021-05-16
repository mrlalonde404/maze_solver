from PIL import Image, ImageDraw
import numpy as np
import pygame
from Maze import *
from Node import *
from Edge import *
from Graph import *
from Colors import *
from collections import deque 
import sys

pygame.init()

# screen size
WIDTH, HEIGHT = 500, 500

# how many frames per second the game should run at
FPS = 1

# global variables
global WIN
global clock
global cells_wide
global cells_high
global maze
global graph
global path


# allows for speed up and slow down of the solving features
def fps_control():
	global FPS
	global pause_fps
	for event in pygame.event.get():
		# stop the game loop
		if event.type == pygame.QUIT:
			quit()

		if pygame.key.get_pressed()[pygame.K_LEFT]:
			FPS = FPS / 2
			if FPS < 0.5:
				FPS = 0.5
			print(f"new FPS: {FPS}")
		if pygame.key.get_pressed()[pygame.K_RIGHT]:
			FPS = FPS * 2
			if FPS > 60:
				FPS = 60
			print(f"new FPS: {FPS}")

def get_nodes_directions(row, col):
		directions = {"left": 0, "up": 0, "right": 0, "down": 0}
		if row == 0:
			# only 1 node on first row, row == 0, the start node, it only has the down direction empty
			directions["down"] = 1
		elif row == maze.get_cells_high() - 1:
			# in the last row the only empty cell only has up available
			directions["up"] = 1
		else:
			# all other nodes not in the first row are not the starting point, 
			# have more than just the down direction with an empty cell
		
			# see if the left spot is empty in the maze
			if maze.get_cells()[row][col - 1] == 0:
				directions["left"] = 1

			# see if the up spot is empty in the maze
			if maze.get_cells()[row-1][col] == 0:
				directions["up"] = 1
			
			# see if the right spot is empty in the maze
			if maze.get_cells()[row][col + 1] == 0:
				directions["right"] = 1
			
			# see if the down spot is empty in the maze
			if maze.get_cells()[row + 1][col] == 0:
				directions["down"] = 1
		return directions


def print_nodes():
	print("All nodes information:")
	for node in graph.get_nodes():
		print(node)


def print_edges():
	print("All edges information:")
	for edge in graph.get_edges():
		print(edge)


# row, col is for a recently added node that we are trying to see if it has an edge to a node above it
def get_above_node(nodes, row, col):
	above_node = None

	# look at above cell for rows >= 1		
	if row >= 1:
		# search from the current row up to the node above this one
		for r in range(row - 1, -1, -1):
			if maze.get_cells()[r][col] == 0: 
				# get the node at this cell, the row-1, col combination
				for i in range(len(nodes) - 1, -1, -1):
					if nodes[i].get_row() == r and nodes[i].get_col() == col:
						above_node = nodes[i]
						break
			elif maze.get_cells()[r][col] == 1:
				break
			if above_node is not None:
				break
	# return the above node, or None if one doesn't exist
	return above_node


# row, col is for a recently added node that we are trying to see if it has an edge to a node to the left of it
def get_left_node(nodes, row, col):
	left_node = None
	
	# look at left cell for columns >= 2
	if col >= 2:
		# print(f"\nNode: row: {row}, col: {col}")
		# search from the current col to the node left of this one
		for c in range(col - 1, -1, -1):
			# print(f"-- row: {row}, c: {c}. cell val: {maze[row][c]}")
			if maze.get_cells()[row][c] == 0: 
				# get the node at this cell, the row, col combination
				for i in range(len(nodes) - 1, -1, -1):
					if nodes[i].get_row() == row and nodes[i].get_col() == c:
						left_node = nodes[i]
						# print(f"match")
						break
			elif maze.get_cells()[row][c] == 1:
				break
			if left_node is not None:
				break
	# return the left node, or None if one doesn't exist
	return left_node


def make_graph():
	global graph
	empty_cell_counter = 0

	# add the nodes, and try to add the edges
	for row in range(len(maze.get_cells())):
		for col in range(len(maze.get_cells()[row])):
			if maze.get_cells()[row][col] == 0:
				# increment the number of empty cells
				empty_cell_counter += 1

				# if a node was added at this row, col combination
				added = False

				# see how many directions there are around the node, if it is a junction or a 
				directions = get_nodes_directions(row, col)
				num_directions = sum(directions.values())

				if num_directions == 1 or num_directions >= 3:
					# only 1 direction, must be connected somehow, so make a node
					# otherwise if there are 3 or 4 possible directions this is an obvious intersection so make a node
					added = True
				elif num_directions == 2:
					if directions["up"] == 1 and directions["left"] == 1:
						added = True
					if directions["up"] == 1 and directions["right"] == 1:
						added = True
					if directions["down"] == 1 and directions["left"] == 1:
						added = True
					if directions["down"] == 1 and directions["right"] == 1:
						added = True

				# if a node at this cell was added, make the edge connections
				if added:
					current_node = Node(row, col, maze.get_cell_size(), WIN)
					graph.add_node(current_node)

					# get all the nodes
					nodes = graph.get_nodes()

					# above and adjacent nodes
					above_node = None
					left_node = None

					# get the above node
					above_node = get_above_node(nodes, row, col)

					if above_node is not None:
						# make the edge, above_node is the start_node, the current_node is the end_node
						edge = Edge(above_node, current_node, WIN)
							
						# add the edge
						graph.add_edge(edge)

					# get the node to the left
					left_node = get_left_node(nodes, row, col)

					if left_node is not None:
						# make the edge, above_node is the start_node, the current_node is the end_node
						edge = Edge(left_node, current_node, WIN)

						# add the edge
						graph.add_edge(edge)
					
				
	node_fill_rate = len(graph.get_nodes()) / (1.0 * empty_cell_counter)
	print(40*"-")
	print(f"Maze empty cells: {empty_cell_counter}")
	print(f"Nodes: {len(graph.get_nodes())}")
	print(f"Node Fill Rate: {node_fill_rate}")
	print(f"Edges: {len(graph.get_edges())}")


# there should only be 1 node on row == 0, the node to start from
def get_start_node(nodes):
	return nodes[0]

# there should only be 1 node on the last row, row == cells_high - 1, this is the final node in the maze, the end point
def get_end_node(nodes):
	return nodes[-1]


def check_if_solved(end_node):
	return end_node.get_visited()


def get_connected_nodes(node):
	# list of the nodes connected to node by the edges in the graph
	connected = []

	for edge in graph.get_edges():
		if edge.get_start_node() == node:
			connected.append(edge.get_end_node())
		elif edge.get_end_node() == node:
			connected.append(edge.get_start_node())

	# return the connected nodes
	return connected


# get all the connected nodes that haven't been visited before
def get_available_nodes(connected):
	available = []
	for node in connected:
		if not node.get_visited():
			available.append(node)
	return available


def get_num_explored_nodes():
	explored = 0
	for n in graph.get_nodes():
		if n.get_visited():
			explored += 1
	return explored	


# if current is None then stop looking at all nodes
def stop_looking_at_other_nodes(current):
	# only the current node should be being looked at
	for node in graph.get_nodes():
		if (node != current and node.get_being_looked_at()) or current is None:
			node.set_being_looked_at(False)

# green nodes are nodes that have not been visited yet
# blue nodes are nodes that have been visited and a dead-end has not been found, and may end up in the final path
# gray nodes are nodes that were visited and led to a dead-end
def depth_first_search(start_node):
	# path of the maze solution
	path = []

	# the current node
	current = start_node

	# while the last node hasn't been visited
	while True:
		# caps the framerate at value of FPS to control how many times this while loop happens per second
		clock.tick(FPS)

		fps_control()

		# mark current visited
		current.visit()

		# add current node to the path
		path.append(current)

		# there is no solution, so send empty list
		if len(path) <= 1 and current.get_removed_from_path():
			return []

		# get the nodes that this current node is connected to from the edges it is found in
		connected = get_connected_nodes(current)

		# get the available nodes from the connected nodes
		available = get_available_nodes(connected)

		# check if the maze has been solved, otherwise,
		# this is a dead-end need to go back up the path until there is options to be taken again
		if len(available) == 0:
			if check_if_solved(get_end_node(graph.get_nodes())):
				break
			else:
				while len(path) > 0:
					# remove the last node from the path
					last = path[-1]
					last.remove()
					path.remove(last)

					# get the previous node in the path
					if len(path) > 0:
						current = path[-1]

					# get the nodes that this current node is connected to from the edges it is found in
					connected = get_connected_nodes(current)

					# get the available nodes from the connected nodes
					available = get_available_nodes(connected)

					# this node now has another path it can take so stop removing nodes
					if len(available) > 0:
						break
		else:
			current = available[0]

		# draw the window to the screen
		draw(draw_graph=True)

	# return the path from the start to the end 
	return path


# the red node is the current node being looked at
# yellow nodes are in the queue and have not been visited yet
# blue nodes have been visited already
# green nodes are not in the queue, have not been visited, and are not being looked at
def breadth_first_search(start, end):
	# make the queue
	queue = deque()

	# the current node, None to start
	current = None

	# add the start node to the queue
	queue.append(start)

	# parent of node i was for all i in length n, the number of nodes
	parents = [None] * maze.get_cells_wide() * maze.get_cells_high()

	# while the queue is not empty
	while len(queue) > 0:
		# caps the framerate at value of FPS to control how many times this while loop happens per second
		clock.tick(FPS)

		fps_control()
		
		# as long as the current node is not the end node
		if current != end:
			# make current node the tail of the queue by removing the element from the end of the queue
			current = queue.pop()
			current.set_in_queue(False)

			# look at the current node
			current.set_being_looked_at(True)
		else:
			break

		# only look at the current node, stop looking at all the other nodes other than current
		stop_looking_at_other_nodes(current)

		# get the nodes that this current node is connected to from the edges it is found in
		connected = get_connected_nodes(current)

		# get the available nodes that haven't been visited from the connected nodes
		available = get_available_nodes(connected)

		# for all available nodes that haven't been visited yet that are connected to the current node
		for node in available:
			if node not in queue:
				# add the node to the queue if
				queue.appendleft(node)

				# mark the node as added to the queue
				node.set_in_queue(True)
					
				# keep track of the parent node of the next node in the parents array
				npos = get_npos(node)
				parents[npos] = current

		# mark current visited
		current.visit()

		# draw the window to the screen
		draw(draw_graph=True)

	# return the parents in the 
	return parents


def get_npos(node):
	return (node.get_row() * maze.get_cells_wide()) + node.get_col()


def make_bfs_path(start, end, parents):
	# path for the maze solution
	path = deque()

	# make the current node by starting at the end of the maze
	current = end

	# reconstruct the path from the beginning to end, starting at the end by appending to the beginning of the path
	while current != None:
			path.appendleft(current)
			at = get_npos(current)
			current = parents[at]

	# if s and e are connected, return the path
	if path[0] == start:
		return path

	# otherwise there is no path, so return an empty list
	return []


# using the weighted dijkstra uses the number of pixels in between the nodes on the image as the cost/weight
# this is the edge length represented as the difference in rows or columns
def dijkstra(start_node, end_node, weighted=True, print_queue=False):
	global path

	# if you want to treat the edges as if every edge has a weight of 1
	if not weighted:
		for edge in graph.get_edges():
			edge.set_length(1)

	# make a queue to hold the nodes
	queue = deque()

	# set the start_node cost to 0
	start_node.set_cost(0)

	# the current node
	current = start_node

	# append the current node to the queue
	queue.append(current)

	while True:
		# caps the framerate at value of FPS to control how many times this while loop happens per second
		clock.tick(FPS)

		fps_control()

		# mark the current node as being looked at
		current.set_being_looked_at(True)

		# only look at the current node, stop looking at all the other nodes other than current
		stop_looking_at_other_nodes(current)

		# get the nodes that this current node is connected to from the edges it is found in
		connected = get_connected_nodes(current)

		# get the available nodes that haven't been visited from the connected nodes
		available = get_available_nodes(connected)

		for node in available:
			# if the unvisited node is not already in the queue, add it to it
			if node not in queue:
				queue.append(node)
				node.set_in_queue(True)

			# get the edge length that both nodes are in 
			length = 0
			for edge in graph.get_edges():
				if (edge.get_start_node() == current and edge.get_end_node() == node) \
				or (edge.get_start_node() == node and edge.get_end_node() == current):
					# get the length of the edge that contains the parent node and the available node
					length = edge.get_length()
					break

			# the nodes current cost is the bigger than the sum of the parent nodes cost and the edge length connecting them
			if node.get_cost() > (current.get_cost() + length):
				# if so then set the nodes cost to the parent nodes cost and the edge length connecting them
				node.set_cost(current.get_cost() + length)

		# mark current visited now that the unvisited neighbors have all been considered
		current.visit()	

		if len(queue) > 0:
			# remove the current node from the queue
			queue.remove(current)
			current.set_in_queue(False)
		else:
			# this is the end node, or there is no solution to the maze
			break

		# show the nodes in the queue
		if print_queue and len(queue) > 0:
			print(40*"-")
			print(f"queue for: ({current.get_row()},{current.get_col()})")
			print(40*"-")
			for node in queue:
				print(node)

		# get the node in the queue that has the smallest cost, set current to this node
		smallest = sys.maxsize
		for node in queue:
			# if the current node is smaller than the running smallest cost, then update the smallest with this nodes cost
			if node.get_cost() < smallest:
				smallest = node.get_cost()
				# keep updating the current node until you get the node that has the smallest cost
				current = node

		# print out the node that was picked with the smallest cost
		if print_queue and len(queue) > 0:
			print(40*"-")
			print(f"picked node: {current}")

		# draw the window to the screen
		draw(draw_graph=True)


	# build the path of the dijkstra maze solution by starting from the end of the maze and working back to the start
	path = deque()

	# start from the last node in the maze
	current = end_node

	# build the path until the start node is reached
	while True:
		# append the current node to the path
		path.appendleft(current)

		# if the cost equals 0 then it is the start node
		if current.get_cost() == 0:
			break

		# get the nodes that share an edge with the current node
		connected = get_connected_nodes(current) 

		# get the connected node that has the smallest cost
		smallest = sys.maxsize
		for node in connected:
			# if the current node is smaller than the running smallest cost, then update the smallest with this nodes cost
			if node.get_cost() < smallest:
				smallest = node.get_cost()
				# keep updating the current node until you get the node that has the smallest cost
				current = node

	# return the path
	return path


def solve_maze(solve_method):
	global path
	path = None

	# get the nodes from the graph
	nodes = graph.get_nodes()

	# the node currently being looked at, start with the starting node
	start = get_start_node(nodes)
	end = get_end_node(nodes)

	while True:
		clock.tick(FPS)
		if path is not None:
			break
		if solve_method == "dfs":
			path = depth_first_search(start)
		elif solve_method == "bfs":
			# get the parents of nodes needed to create the path from searching the entire graph
			parents = breadth_first_search(start, end)
			# use the start, end, and parents to construct the path solution for this maze
			path  = make_bfs_path(start, end, parents)
		elif solve_method == "dijkstra":
			# changing weighted from True to False almost makes dijkstra identical to bfs
			path = dijkstra(start, end,  weighted=True, print_queue=True)
		# light the path up in red
		show_path(path, print_path=False)
		# draw the window and all of its elements to the screen
		draw(draw_graph=True)

	print(40*"-")
	if len(path) != 0:
		# print information about the path
		print(f"Path length: {len(path)}")
		print(f"Nodes explored: {get_num_explored_nodes()}")
		print(f"Nodes explored rate: {get_num_explored_nodes() * 1.0 / len(graph.get_nodes())}")
	else:
		print("Solver couldn't find a solution")


def show_path(path, print_path=False):
	# stop looking at any node being looked at
	stop_looking_at_other_nodes(current=None)
	# show the final solution of the path by marking the solution nodes RED by setting the being_looked_at flags to True  
	for node in path:
		node.set_being_looked_at(True)
		if print_path:
			print(40*"-")
			print(node)

def setup():
	global cells_wide
	global cells_high
	global maze
	global graph

	# if the maze attributes have been set, then create the cells list in the maze
	if maze.get_cells_wide() > 0 and maze.get_cells_high() > 0:
		# global maze for creation, exporting, and solving
		m = [(maze.get_cells_wide() * [0]) for _ in range(maze.get_cells_high())]

		# turn maze into a numpy array, only store values from 0 to 255
		maze.set_cells(np.array(m, dtype=np.uint8))

	# make the graph for building and solving later
	graph = Graph()

# mpos is the input mouse position
def get_row_col_from_mouse(mpos):
	mx, my = mpos
	# use integer division to get the row and col that the cell is in using the cell size
	return (my // maze.get_cell_size()), (mx // maze.get_cell_size()) 


def draw(draw_graph=False):
	# fill the screen with a white background to reset it
	WIN.fill(WHITE)

	# draw the maze
	maze.draw_maze()		

	# draw the graph accordingly
	if draw_graph:
		graph.draw_graph()

	# update the screen to show changes
	pygame.display.update()


# load a maze from the input file
def load_maze(input_file):
	global maze
	global WIDTH
	global HEIGHT

	# try to load in the image within the folder named input_file
	try:
		img = Image.open(input_file)
	except Exception:
		print(f"Failed to load: {input_file}")
		quit()

	# set the maze width and height
	maze.set_cells_wide(img.size[0])
	maze.set_cells_high(img.size[1])

	# scale the cell_size
	maze.scale_cell_size()

	# global maze for creation, exporting, and solving
	m = [(maze.get_cells_wide() * [0]) for _ in range(maze.get_cells_high())]

	# turn maze into a numpy array, only store values from 0 to 255
	maze.set_cells(np.array(m, dtype=np.uint8))

	# fix the width and height
	WIDTH, HEIGHT = img.size[0] * maze.get_cell_size(), img.size[1] * maze.get_cell_size()

	# load the maze in now so that it can be displayed
	for row in range(img.width):
		for col in range(img.width):
			# the val here will be a RGB color, get it in the form (col, row)-> (x,y)
			val = img.getpixel((col, row))

			# get the corresponding value to the RGB color
			if val == WHITE:
				val = 0
			elif val == BLACK:
				val = 1
			maze.get_cells()[row][col] = val
	print(40*"-")
	print(f"Image loaded from {input_file}")


def export_maze(file_name):
	# create the image
	img = Image.new(mode="RGB", size=(maze.get_cells_wide(), maze.get_cells_high()))

	# fill the image according to the maze
	for row in range(len(maze.get_cells())):
		for col in range(len(maze.get_cells()[row])):
			if maze.get_cells()[row][col] == 0:
				img.putpixel((col, row), WHITE)

			elif maze.get_cells()[row][col] == 1:
				img.putpixel((col, row), BLACK)
			
	# save the image to the file_name as a PNG
	img.save(file_name, "PNG")
	print(40*"-")
	print(f"Output image resolution: {maze.get_cells_wide()}, {maze.get_cells_high()}")
	print("Image exported from maze created by user")


def export_solved_maze(file_name, solve_method):
	# create the solution image
	img = Image.new(mode="RGB", size=(maze.get_cells_wide(), maze.get_cells_high()))

	# get the nodes and edges from the graph
	nodes = graph.get_nodes()
	edges = graph.get_edges()

	# fill the image according to the maze
	for row in range(len(maze.get_cells())):
		for col in range(len(maze.get_cells()[row])):
			if maze.get_cells()[row][col] == 0:
				img.putpixel((col, row), WHITE)

			elif maze.get_cells()[row][col] == 1:
				img.putpixel((col, row), BLACK)

	# for every node in the path
	incr = int((255*2.0) / len(path))

	# r, g, b channel values
	r = 0
	g = 0
	b = 255

	# go over every node and use the increment on the r, g, b values to lineraly interpolate the solution path
	for i in range(len(path)):
		# linearl interpolate the r,g b values
		if r < 255:
			r += incr
		if r >= 255:
			r = 255
			b -= incr
		if b < 0:
			b = 0 

		# fill in the path
		img.putpixel((path[i].get_col(), path[i].get_row()), (r, g, b))

		# if not the last node, fill in the edges
		if i != len(path) - 1:
			# get the next node in the path, the one connected from the current cell by the edge
			next_node = path[i + 1]

			# if the nodes are on the same row
			if path[i].get_row() == next_node.get_row():
				if path[i].get_col() < next_node.get_col():
					for col in range(path[i].get_col(), next_node.get_col()+1):
						img.putpixel((col, path[i].get_row()), (r, g, b))
				elif path[i].get_col() > next_node.get_col():
					for col in range(path[i].get_col(), next_node.get_col() - 1, -1):
						img.putpixel((col, path[i].get_row()), (r, g, b))

			# if the nodes are on the same column
			elif path[i].get_col() == next_node.get_col():
				if path[i].get_row() < next_node.get_row():
					for row in range(path[i].get_row(), next_node.get_row()+1):
						img.putpixel((path[i].get_col(), row), (r, g, b))
				elif path[i].get_row() > next_node.get_row():
					for row in range(path[i].get_row(), next_node.get_row() -1, -1):
						img.putpixel((path[i].get_col(), row), (r, g, b))

	# save the image to the export_file_name as a PNG
	export_file_name = file_name + solve_method + ".png"
	img.save(export_file_name, "PNG")
	print(40*"-")
	print(f"Output image resolution: {maze.get_cells_wide()}, {maze.get_cells_high()}")
	print(f"Saved solution image to {export_file_name}")


def main():
	global WIDTH
	global HEIGHT
	global clock
	global WIN
	global maze

	# make a clock object to control FPS
	clock = pygame.time.Clock()

	# make the maze object 
	maze = Maze()

	# bool to keep track of when the game loop should stop running
	run = True

	# if the user is going to be making the maze
	make_maze = False

	# if the image was saved
	saved = False

	# keep working on the maze if it is valid and pressed enter to save but the user wants to make changes
	keep_working = False

	# if the user is not going to watch the maze be solved, true if they are
	draw_graph = False

	# when the user isn't making a maze, solve it
	solve = False

	# method to solve the maze
	solve_method = ""

	# solving has been finished
	finished = False

	# reset and try a different solver
	reset = False

	# see if the maze should be made
	while True:
		print(40*"-")
		selection = input("Make maze (y/n)? ")
		if selection.lower() == "y" or selection.lower() == "n":
			break

	if selection.lower() == "y":
		# user will be making the maze
		make_maze = True

		print(40*"-")
		print("Enter size of square maze,")
		print("will have the same width and height, ")
		size = input(f"must be {maze.get_min_num_cells()} <= size <= {maze.get_max_num_cells()}): ")
		while True:
			try:
				size = int(size)

				if size > maze.get_max_num_cells(): 
					size = maze.get_max_num_cells()
				elif size <= maze.get_min_num_cells(): 
					size = maze.get_min_num_cells()

				# set the maze width and height
				maze.set_cells_wide(size)
				maze.set_cells_high(size)

				# scale the cell size according the size of the maze
				maze.scale_cell_size()

				# scaling width and height
				WIDTH = HEIGHT = size * maze.get_cell_size()

				print(40*"-")
				print("Press the \"Enter\" key to save the maze:")
				break
			except Exception:
				print("Invalid size entered")
		

	# setup the pygame objects
	setup()

	# if not making the maze
	if selection.lower() == "n":
		# get the input file from the user for the maze
		file = input("Enter name of maze file to solve: ")

		# load the maze into the program
		load_maze(file)

		# user is going to watch the graph be drawn too, and then solved
		draw_graph = True
		solve = True

	# make the pygame display
	WIN = pygame.display.set_mode((WIDTH, HEIGHT))
	pygame.display.set_caption("Maze Solver")

	maze.set_WIN(WIN)

	# game loop
	while run:
		# caps the framerate at value of FPS to control how many times this while loop happens per second
		clock.tick(FPS)

		# loop through all game events
		for event in pygame.event.get():
			# stop the game loop
			if event.type == pygame.QUIT:
				if make_maze and not saved:
					print("The window was closed prior to saving the maze")
				run = False

			if make_maze:
				# look at mouse events
				if pygame.mouse.get_pressed()[0]: 
					# left mouse button pressed, get its position
					mpos = pygame.mouse.get_pos()

					# get the row, col of where the mouse is
					row, col = get_row_col_from_mouse(mpos)

					# set this cell to black -> change maze cell at (row,col)->(y,x) to 1
					maze.get_cells()[row][col] = 1

				elif pygame.mouse.get_pressed()[2]: 
					 # right mouse button pressed, get its positions
					mpos = pygame.mouse.get_pos()

					# get the row, col of where the mouse is
					row, col = get_row_col_from_mouse(mpos)

					# reset this cell to white -> change maze cell at (row,col)->(y,x) to 0
					maze.get_cells()[row][col] = 0

				# see if the maze should be saved or not
				if pygame.key.get_pressed()[pygame.K_RETURN]:
					# if the user continue to work and pressed enter to prompt the save menu again, turn off the keep_working flag
					if keep_working:
						keep_working = False
					if not keep_working:
						# check if the maze is valid and if it is then prompt the user with the option to save it
						if maze.check_if_valid_maze():
							print(40*"-")
							print("You must save a maze in order to solve it")
							# ask the user if they want to save the maze if they press then enter key while making the maze
							while not saved and not keep_working:
								print(40*"-")
								save = input("Would you like to save the maze (y/n)? ")
								if save.lower() == "y":
									# export the maze into a PNG file
									out = "maze" + str(size) + ".png"
									export_maze(out)
									saved = True
									break
								elif save.lower() == "n":
									while True:
										print(40*"-")
										keep = input("Would you like to keep working on the maze (y/n)? ")
										if keep.lower() == "y":
											keep_working = True
											break
										elif keep.lower() == "n":
											run = False
											break
							# see if the user would like to watch the solver solve the maze they just made
							if saved and not keep_working:
								while True:
									print(40*"-")
									sol = input("Would you like to solve the maze (y/n)? ")
									if sol.lower() == "y":
										draw_graph = True
										solve = True
										break
									elif sol.lower() == "n":
										run = False
										break

		# if solve is selected, choose method to solve
		if solve and not finished:
			# after the method has been selected, solve the maze with it
			if len(graph.get_nodes()) == 0 or reset:
				# if a reset has already eben done before, flip the reset flag back to False
				if reset:
					reset = False
				if len(graph.get_nodes()) == 0:
					make_graph()

			# draw the window to the screen
			draw(draw_graph)
		
			# get the method for how to solve the maze
			options = {1: "dfs", 2: "bfs", 3: "dijkstra"}
			print(40*"-")
			print("Methods to solve the maze:")
			for k, v in options.items():
				print(f"{k}. {v}")
			
			while True:
				print(40*"-")
				selection = input("Choose a method to solve the maze: ")
				try:
					selection = int(selection)
					if selection in options.keys():
						solve_method = options[selection]
						print(40*"-")
						print(f"Chose method: {solve_method}")
						print(f"Current FPS: {FPS}")
						break
					else:
						print("Invalid option.")
				except Exception:
					print("Invalid type.")
				
			# solve the maze now that the graph has been made
			solve_maze(solve_method)

			# if there was a solution, then the solving is finished
			if len(path) > 0:
				finished = True
			if solve and finished:
				out = "maze" + str(maze.get_cells_wide()) + "_solved_"
				export_solved_maze(out, solve_method)
			else:
				# can't be solved so stop trying to
				solve = False

		# see if the user would like to try solving it with another solve_method
		if solve and finished:
			while True:
				print(40*"-")
				res = input("Would you like to try another method of solving the maze (y/n)? ")
				if res.lower() == "y":
					reset = True
					break
				elif res.lower() == "n":
					run = False
					break

			# if the user wants to try another method of solving the maze
			if reset:
				graph.reset_nodes()
				solve_method = ""
				finished = False

		# draw the window to the screen
		draw(draw_graph)

	# quit the display for pygame
	pygame.quit()


if __name__ == "__main__":
	main()
