import pygame 
from Colors import *

class Maze:
	def __init__(self):
		# how many cells wide and high the maze will be
		self.cells_wide = 0
		self.cells_high = 0

		# all the maze cells
		self.cells = []
		
		# how wide and tall each cell in the maze is
		self.cell_size = 25
		self.min_cell_size = 25
		self.max_cell_size = 80

		# pixel limitations for creating mazes
		self.min_num_cells = 10
		self.max_num_cells = 40

		# the window to draw on
		self.WIN = None

	def scale_cell_size(self):
		# scaling cell_size according to the size and width of the maze
		cell_scale = self.cells_wide // 4
		self.cell_size = ((self.max_num_cells // 4) - cell_scale) * 10
		
		# use limits on cell_size to constrain its ability to scale
		if self.cell_size <= self.min_cell_size:
			self.cell_size = self.min_cell_size
		
		elif self.cell_size >= self.max_cell_size:
			self.cell_size = self.max_cell_size

	def draw_maze(self):
		# fill in the cells with the approprate colors according to the maze 2D array
		for row in range(len(self.cells)):
			for col in range(len(self.cells[row])):
				# get the color based on the number in the maze at row, col
				if self.cells[row][col] == 0:
					c = WHITE
				elif self.cells[row][col] == 1:
					c = BLACK

				# get the x,y of the top left corner of the cell
				sx = col * self.cell_size
				sy = row * self.cell_size

				# draw each cell using the row, col, and cell size
				pygame.draw.rect(self.WIN, rect=(sx, sy, sx + self.cell_size, sy + self.cell_size), color=c)

		# draw the grid lines
		self.draw_grid_lines()


	def draw_grid_lines(self):
	    # draw the horizontal lines first from the left of the screen to the right(the width)
	    for i in range(self.cells_high):
	        pygame.draw.line(self.WIN, GRAY, (0, i * self.cell_size), (self.WIN.get_width(), i * self.cell_size))

	    # draw the vertical lines from the top of the screen to the bottom(the height)
	    for j in range(self.cells_wide):
	        pygame.draw.line(self.WIN, GRAY, (j * self.cell_size, 0), (j * self.cell_size, self.WIN.get_height()))

	def get_cells_wide(self):
		return self.cells_wide

	def get_cells_high(self):
		return self.cells_high

	def get_cells(self):
		return self.cells

	def get_cell_size(self):
		return self.cell_size

	def get_min_num_cells(self):
		return self.min_num_cells

	def get_max_num_cells(self):
		return self.max_num_cells

	def set_cells_wide(self, cells_wide):
		self.cells_wide = cells_wide

	def set_cells_high(self, cells_high):
		self.cells_high = cells_high

	def set_cells(self, cells):
		self.cells = cells

	def set_WIN(self, WIN):
		self.WIN = WIN

	# this checks if there is only 1 open cell in the top row and in the bottom row, 
	# and that both the left and right columns are completely filled
	# can't solve an invalid maze, so this will return True or False 
	# and detail why it can't be solved
	def check_if_valid_maze(self):
		top_row_count = 0
		bottom_row_count = 0
		left_col_count = 0
		right_col_count = 0
		invalid = False

		# count the number of empty spaces in the top row and the bottom row
		for col in range(self.cells_wide):
			# count the top row empty cells
			row = 0
			if self.cells[row][col] == 0:
				top_row_count += 1

			# count the bottom row empty cells
			row = self.cells_high - 1
			if self.cells[row][col] == 0:
				bottom_row_count += 1

		# count the number of empty spaces in the left and right columns
		for row in range(self.cells_high):
			# count the left column empty cells
			col = 0
			if self.cells[row][col] == 0:
				left_col_count += 1

			# count the bottom row empty cells
			row = self.cells_wide - 1
			if self.cells[row][col] == 0:
				right_col_count += 1

		# check all the counts for if it is valid
		print(40*"-")
		print("Checking if maze is valid:")
		print(40*"-")
		print(f"Top row empty spaces count: {top_row_count}")
		print(f"Bottom row empty spaces count: {bottom_row_count}")
		print(f"Left column empty spaces count: {left_col_count}")
		print(f"Right column empty spaces count: {right_col_count}")
		print(40*"-")

		if top_row_count != 1:
			invalid = True
			diff = top_row_count - 1
			print(f"The top row must contain 1 empty cell, the start point. Fill in: {diff} cells")

		if bottom_row_count != 1:
			invalid = True
			diff = bottom_row_count - 1
			print(f"The bottom row must contain 1 empty cell, the end point. Fill in: {diff} cells")
		
		if left_col_count > 0:
			invalid = True
			diff = self.cells_high - left_col_count
			print(f"The left column must have every cell filled in. Fill in: {diff} cells")
		
		if right_col_count > 0:
			invalid = True
			diff = self.cells_high - right_col_count
			print(f"The right column must have every cell filled in. Fill in: {diff} cells")

		# check all conditions to see if maze is valid and return outcome
		if invalid:
			print(40*"-")
		if top_row_count == 1 and bottom_row_count == 1 and left_col_count == 0 and right_col_count == 0:
			print("Maze is valid")
			return True
		else:
			print("Maze is invalid for reasons listed above")
			return False