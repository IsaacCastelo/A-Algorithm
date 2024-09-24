import heapq
import matplotlib.pyplot as plt
import numpy as np

def a_star(start, goal, h, neighbors, grid):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: h(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in neighbors(current, grid):
            tentative_g_score = g_score[current] + 1  

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + h(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No encontro camino

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def chebyshev_distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

def neighbors(node, grid):
    x, y = node
    potential_neighbors = [(x+dx, y+dy) for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]]
    valid_neighbors = []
    for nx, ny in potential_neighbors:
        if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and grid[nx, ny] != 1:
            valid_neighbors.append((nx, ny))
    return valid_neighbors

def plot_path(path, grid, start, goal):
    display_grid = np.full(grid.shape, ' ')
    for (x, y) in path:
        display_grid[x, y] = 'X'
    display_grid[goal] = 'O'
    display_grid[start] = 'S'

    fig, ax = plt.subplots()
    ax.matshow(np.where(grid == 1, 0.7, 1), cmap='Greys')

    for (i, j), val in np.ndenumerate(display_grid):
        ax.text(j, i, val, ha='center', va='center', color='red')

    plt.show()

# Example usage:
grid = np.array([
    [0, 0, 0, 0, 0, 0],
    [1, 0, 1, 1, 0, 0],
    [0, 1, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0]
])
start = (0, 0)
goal = (5, 5)
path = a_star(start, goal, chebyshev_distance, neighbors, grid)
print("Path:", path)
plot_path(path, grid, start, goal)
