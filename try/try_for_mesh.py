import numpy as np

ex_verts = np.array([
    [245.5097, 366.4517, 75.0000],
     [290.6382, 28.5419, 75.0000],
    [243.8408, 63.2476, 111.7922],
    [240.9064, 79.8072, 112.4230],
    [199.7914, 111.0648, 144.6584],
    [177.2708, 276.7621, 145.0000],
     [207.5955, 17.4195, 75.0000],
    [204.2353, 32.3258, 75.0000],
    [153.4020, 294.4090, 126.9071],
    [129.5602, 22.4224, 75.0000],
    [101.2336, 227.1593, 75.0000],
    [79.5783, 284.2168, 127.2086],
    [31.0913, 217.1453, 75.0000],
    [15.2045, 335.7384, 75.0000],
    [245.5097, 366.4517, 0.0],
     [290.6382, 28.5419, 0.0],
     [207.5955, 17.4195, 0.0],
    [204.2353, 32.3258, 0.0],
    [129.5602, 22.4224, 0.0],
    [101.2336, 227.1593, 0.0],
    [31.0913, 217.1453, 0.0],
     [15.2045, 335.7384, 0.0]
])

ex_faces = [
    [9, 4, 3, 7],
    [7, 6, 2, 3],
    [6, 1, 2],
    [1, 0, 5, 4, 3, 2],
    [0, 13, 11, 8, 5],
    [11, 12, 13],
    [12, 10, 8, 11],
    [10, 9, 4, 5, 8],
    [14, 15, 16, 17, 18, 19, 20, 21],
    [0, 14, 15, 1],
     [1, 15, 16, 6],
    [6, 16, 17, 7],
    [7, 17, 18, 9],
    [9, 18, 19, 10],
    [10, 19, 20, 12],
    [12, 20, 21, 13],
     [13, 21, 14, 0]
]


def is_point_on_segment(px, py, x1, y1, x2, y2, tol=1e-8):
    cross_product = (py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)
    if abs(cross_product) > tol:
        return False
    min_x = min(x1, x2) - tol
    max_x = max(x1, x2) + tol
    min_y = min(y1, y2) - tol
    max_y = max(y1, y2) + tol
    return (px >= min_x and px <= max_x) and (py >= min_y and py <= max_y)


def point_in_polygon(px, py, polygon, tol=1e-8):
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if is_point_on_segment(px, py, x1, y1, x2, y2, tol):
            return True
    inside = False
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > py) != (y2 > py):
            dy = y2 - y1
            if abs(dy) < tol:
                continue
            t = (py - y1) / dy
            x_intersect = x1 + t * (x2 - x1)
            if px <= x_intersect + tol:
                inside = not inside
    return inside


def compute_normal(vertices):
    for i in range(len(vertices) - 2):
        p0 = np.array(vertices[i])
        p1 = np.array(vertices[i + 1])
        p2 = np.array(vertices[i + 2])
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            return normal / norm
    return None


def process_face(face):
    face_verts = [ex_verts[i].tolist() for i in face]
    normal = compute_normal(face_verts)
    if normal is None:
        return []

    face_z = [v[2] for v in face_verts]
    min_z = min(face_z)
    edges = []
    n = len(face)
    for i in range(n):
        current = face[i]
        next_idx = face[(i + 1) % n]
        a = ex_verts[current]
        b = ex_verts[next_idx]
        edges.append((a, b))

    candidate_edges = [(a, b) for a, b in edges if a[2] == min_z and b[2] == min_z]
    if not candidate_edges:
        return []

    A = np.array(candidate_edges[0][0])
    B = np.array(candidate_edges[0][1])
    u_vector = B - A
    u_norm = np.linalg.norm(u_vector)
    if u_norm < 1e-6:
        return []
    u_vector = u_vector / u_norm

    v_vector = np.cross(normal, u_vector)
    v_norm = np.linalg.norm(v_vector)
    if v_norm < 1e-6:
        return []
    v_vector = v_vector / v_norm

    uv_coords = []
    for idx in face:
        vertex = np.array(ex_verts[idx])
        rel_vec = vertex - A
        u = np.dot(rel_vec, u_vector)
        v = np.dot(rel_vec, v_vector)
        uv_coords.append((u, v))

    u_coords = [u for u, v in uv_coords]
    v_coords = [v for u, v in uv_coords]
    u_min, u_max = min(u_coords), max(u_coords)
    v_min, v_max = min(v_coords), max(v_coords)

    grid_size = 15.0
    squares = []
    current_u = u_min
    while current_u < u_max:
        current_v = v_min
        while current_v < v_max:
            square_u_end = min(current_u + grid_size, u_max)
            square_v_end = min(current_v + grid_size, v_max)
            # Define square points: corners, edge midpoints, and center
            corners = [
                (current_u, current_v),
                (square_u_end, current_v),
                (square_u_end, square_v_end),
                (current_u, square_v_end)
            ]
            mid_top = ((current_u + square_u_end) / 2, current_v)
            mid_bottom = ((current_u + square_u_end) / 2, square_v_end)
            mid_left = (current_u, (current_v + square_v_end) / 2)
            mid_right = (square_u_end, (current_v + square_v_end) / 2)
            center = ((current_u + square_u_end) / 2, (current_v + square_v_end) / 2)
            check_points = corners + [mid_top, mid_bottom, mid_left, mid_right, center]

            all_inside = True
            for (u, v) in check_points:
                if not point_in_polygon(u, v, uv_coords):
                    all_inside = False
                    break
            if all_inside:
                squares.append((current_u, current_v, square_u_end, square_v_end))
            current_v += grid_size
        current_u += grid_size

    mesh_squares = []
    for (start_u, start_v, end_u, end_v) in squares:
        corners = [
            (start_u, start_v),
            (end_u, start_v),
            (end_u, end_v),
            (start_u, end_v)
        ]
        square_3d = []
        for u, v in corners:
            displacement = u * u_vector + v * v_vector
            point_3d = A + displacement
            square_3d.append(point_3d.tolist())
        mesh_squares.append(square_3d)

    return mesh_squares







