"""
dijkstra_visual.py
Interactive visual simulation of Dijkstra's algorithm using OpenCV by Sourbh Kumar.

Dependencies:
    pip install numpy opencv-python

Usage:
    - Left-click and drag to draw/remove walls (toggle).
    - Press 's' then left-click to set Start node.
    - Press 'e' then left-click to set End node.
    - Press SPACE to run the algorithm.
    - Press 'p' to pause/resume during run.
    - Press 'r' to reset grid (start/end cleared).
    - Press 'c' to clear only walls (keep start/end).
    - Press 'q' or ESC to quit.
"""

import cv2
import numpy as np
import heapq
import time

# ---------- Configuration ----------
CELL_SIZE = 24          # pixels per grid cell
GRID_W = 28             # columns
GRID_H = 18             # rows
WINDOW_NAME = "Dijkstra Visual - Left-drag:walls | s: set start | e: set end | SPACE: run"
DELAY = 1               # ms in waitKey between frame updates (1 makes smooth animation)
DIAGONAL = False        # allow diagonal moves (cost sqrt2) if True
# -----------------------------------

# Colors (BGR for OpenCV)
COLOR_BG = (30, 30, 30)
COLOR_GRID = (50, 50, 50)
COLOR_EMPTY = (45, 45, 45)
COLOR_WALL = (20, 20, 120)
COLOR_START = (30, 220, 30)
COLOR_END = (30, 180, 220)
COLOR_FRONTIER = (0, 165, 255)   # orange
COLOR_VISITED = (200, 50, 50)    # red-ish
COLOR_PATH = (40, 230, 40)
COLOR_TEXT = (220, 220, 220)

W = GRID_W * CELL_SIZE
H = GRID_H * CELL_SIZE

# Grid states
EMPTY = 0
WALL = 1

# Initialize grid
grid = np.zeros((GRID_H, GRID_W), dtype=np.uint8)

start = None  # (r, c)
end = None    # (r, c)

mouse_down = False
placing_wall = True  # toggle mode for dragging
mode = None  # 's' for placing start, 'e' for placing end, None for drawing walls

paused = False

# For drawing visualization
frontier_set = set()
visited_set = set()
parent = dict()
dist = dict()
found_shortest = False


def grid_to_pixel(cell):
    r, c = cell
    x = c * CELL_SIZE
    y = r * CELL_SIZE
    return x, y


def pixel_to_grid(x, y):
    c = x // CELL_SIZE
    r = y // CELL_SIZE
    if 0 <= r < GRID_H and 0 <= c < GRID_W:
        return (r, c)
    return None


def draw_grid(img):
    # background
    img[:] = COLOR_BG

    # cells
    for r in range(GRID_H):
        for c in range(GRID_W):
            x, y = c * CELL_SIZE, r * CELL_SIZE
            rect = (x + 1, y + 1, CELL_SIZE - 2, CELL_SIZE - 2)
            state = grid[r, c]
            if state == WALL:
                color = COLOR_WALL
            else:
                color = COLOR_EMPTY
            cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), color, thickness=-1)

    # frontier, visited
    for cell in visited_set:
        x, y = grid_to_pixel(cell)
        cv2.rectangle(img, (x + 2, y + 2), (x + CELL_SIZE - 3, y + CELL_SIZE - 3), COLOR_VISITED, -1)
    for cell in frontier_set:
        x, y = grid_to_pixel(cell)
        cv2.rectangle(img, (x + 3, y + 3), (x + CELL_SIZE - 4, y + CELL_SIZE - 4), COLOR_FRONTIER, -1)

    # path
    if parent and (end in parent or found_shortest):
        # reconstruct path
        cur = end
        path_cells = []
        while cur in parent:
            path_cells.append(cur)
            cur = parent[cur]
        # include start
        if cur == start:
            path_cells.append(start)
            for cell in path_cells:
                x, y = grid_to_pixel(cell)
                cv2.rectangle(img, (x + 4, y + 4), (x + CELL_SIZE - 5, y + CELL_SIZE - 5), COLOR_PATH, -1)

    # start / end
    if start:
        x, y = grid_to_pixel(start)
        cv2.rectangle(img, (x + 1, y + 1), (x + CELL_SIZE - 1, y + CELL_SIZE - 1), COLOR_START, -1)
    if end:
        x, y = grid_to_pixel(end)
        cv2.rectangle(img, (x + 1, y + 1), (x + CELL_SIZE - 1, y + CELL_SIZE - 1), COLOR_END, -1)

    # grid lines
    for i in range(GRID_W + 1):
        cv2.line(img, (i * CELL_SIZE, 0), (i * CELL_SIZE, H), COLOR_GRID, 1)
    for i in range(GRID_H + 1):
        cv2.line(img, (0, i * CELL_SIZE), (W, i * CELL_SIZE), COLOR_GRID, 1)

    # overlay text
    info = f"Start: {start}  End: {end}   Mode: {'placing start' if mode=='s' else ('placing end' if mode=='e' else 'draw walls')}"
    cv2.putText(img, info, (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)


def neighbors(cell):
    r, c = cell
    nbrs = []
    # 4-neighborhood
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if DIAGONAL:
        moves += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dr, dc in moves:
        nr, nc = r + dr, c + dc
        if 0 <= nr < GRID_H and 0 <= nc < GRID_W:
            if grid[nr, nc] != WALL:
                # cost: 1 for orthogonal, sqrt(2) for diagonal
                cost = 1.0
                if DIAGONAL and abs(dr) + abs(dc) == 2:
                    cost = 2 ** 0.5
                nbrs.append(((nr, nc), cost))
    return nbrs


def dijkstra_visual(start_cell, end_cell, draw_callback):
    """
    Generator-based Dijkstra so we can visualize step by step.
    Yields after popping each node (for animation).
    """
    global frontier_set, visited_set, parent, dist, found_shortest
    frontier_set = set()
    visited_set = set()
    parent = {}
    dist = {}
    found_shortest = False

    pq = []
    dist[start_cell] = 0.0
    heapq.heappush(pq, (0.0, start_cell))
    frontier_set.add(start_cell)

    while pq:
        dcur, cur = heapq.heappop(pq)
        # if we popped an out-of-date distance, skip
        if dcur > dist.get(cur, float('inf')):
            continue

        # mark visited
        frontier_set.discard(cur)
        visited_set.add(cur)

        # yield state so caller can draw
        yield cur, dcur, False  # not finished

        if cur == end_cell:
            found_shortest = True
            break

        for (nb, w) in neighbors(cur):
            nd = dcur + w
            if nd < dist.get(nb, float('inf')):
                dist[nb] = nd
                parent[nb] = cur
                heapq.heappush(pq, (nd, nb))
                frontier_set.add(nb)

        # small sleep to slow animation (stable across machines)
        time.sleep(0.01)

    # final yield to allow path draw
    yield end_cell, dist.get(end_cell, float('inf')), True


# Mouse callback
def on_mouse(event, x, y, flags, param):
    global mouse_down, placing_wall, start, end, mode
    cell = pixel_to_grid(x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_down = True
        if mode == 's':
            if cell and grid[cell[0], cell[1]] != WALL:
                start = cell
            mode = None
        elif mode == 'e':
            if cell and grid[cell[0], cell[1]] != WALL:
                end = cell
            mode = None
        else:
            # toggle wall: if cell is wall, remove; else add
            if cell:
                r, c = cell
                if grid[r, c] == WALL:
                    grid[r, c] = EMPTY
                    placing_wall = False
                else:
                    grid[r, c] = WALL
                    placing_wall = True
    elif event == cv2.EVENT_MOUSEMOVE and mouse_down and mode is None:
        # draw while dragging
        if cell:
            r, c = cell
            grid[r, c] = WALL if placing_wall else EMPTY
    elif event == cv2.EVENT_LBUTTONUP:
        mouse_down = False


def main():
    global mode, paused, start, end, frontier_set, visited_set, parent, found_shortest

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    img = np.zeros((H, W, 3), dtype=np.uint8)

    running = True
    algorithm_running = False
    algo_gen = None

    while running:
        draw_grid(img)
        cv2.imshow(WINDOW_NAME, img)

        key = cv2.waitKey(DELAY) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or q
            running = False
            break
        elif key == ord('s'):
            mode = 's'
        elif key == ord('e'):
            mode = 'e'
        elif key == ord('c'):
            # clear only walls
            grid[:] = EMPTY
            frontier_set.clear()
            visited_set.clear()
            parent.clear()
            found_shortest = False
            algorithm_running = False
            algo_gen = None
        elif key == ord('r'):
            # reset everything
            grid[:] = EMPTY
            start = None
            end = None
            frontier_set.clear()
            visited_set.clear()
            parent.clear()
            found_shortest = False
            algorithm_running = False
            algo_gen = None
        elif key == ord('p'):
            paused = not paused
        elif key == ord(' '):
            # run Dijkstra if start and end set
            if start is None or end is None:
                print("Please set both start and end nodes before running.")
            else:
                algorithm_running = True
                paused = False
                algo_gen = dijkstra_visual(start, end, draw_grid)
        # if algorithm running, step it
        if algorithm_running and algo_gen is not None and not paused:
            try:
                cur, distcur, finished = next(algo_gen)
                # update display after each step (draw_grid will be called next loop)
                if finished:
                    # reconstruct path (parent filled)
                    algorithm_running = False
                    print("Finished. Distance:", dist.get(end, float('inf')))
            except StopIteration:
                algorithm_running = False
                algo_gen = None

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
