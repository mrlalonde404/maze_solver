# maze_solver

# uses pygame to make a maze creation interface that has rules for valid mazes
# - a valid maze must have 1 empty cell on the top and bottom row, and the entire left and right column's cells filled in

# green nodes are cells that have not been visited, blue have been visited, yellow are in a queue

# edges are blue when both nodes in the edge have been visited, and are red when only 1 or neither have been visited

# implements depth first search: gray nodes were visited and then removed from the path

# implements breadth first search: the red node is the node currenly being looked at

# implements dijkstra's shortest path algorithm: when the weight option is set to True it uses the difference in rows or columns to for edge length and therefore affects the cost associated with every node, otherwise if it is set to False it is almost BFS. If print_queue set to True then it shows how the alogrithm is selecting the next node to be looked at and how the minimum cost is being selected

# when making a maze the enter key is used to save the maze after it is deemed to be valid

# when a maze is being solved the left arrow key can be pressed to reduce the FPS by half and has a lower bound of 0.5 FPS, whereas the the right arrow key can be pressed to double the FPS and has an upper bound of 30 FPS

# a solved maze will be saved into a PNG with the path linearly interpolated from blue to red with the solved maze having the solve method that was used in the PNG name

# if a maze cannot be solved the program will tell the user in the console that it can't be and will continue to display the maze until the window is closed
