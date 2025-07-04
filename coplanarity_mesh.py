from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import read_polyshape_3d
import numpy as np
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Polygon
import matplotlib.patches as patches

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Polygon
import matplotlib.patches as patches


class FaceProjectionVisualizer:
    """
    Visualizes the 2D projection of roof faces and grid generation process
    """

    def __init__(self, roof_solar_panel):
        self.roof = roof_solar_panel

    def visualize_face_projection(self, face_index=0, grid_size=None):
        """
        Visualize the 2D projection of a specific roof face

        Args:
            face_index: Index of the roof face to visualize
            grid_size: Grid size for mesh generation (uses roof's grid_size if None)
        """
        if face_index >= len(self.roof.roof_faces):
            print(f"Face index {face_index} out of range. Available faces: 0-{len(self.roof.roof_faces) - 1}")
            return

        face = self.roof.roof_faces[face_index]
        if grid_size is None:
            grid_size = self.roof.grid_size

        projection_data = self._get_face_projection_data(face, grid_size)

        if projection_data is None:
            print(f"Could not process face {face_index}")
            return

        self._plot_2d_projection(projection_data, face_index)

    def visualize_all_faces(self, grid_size=None):
        """
        Visualize 2D projections of all roof faces in a grid layout
        """
        if grid_size is None:
            grid_size = self.roof.grid_size

        num_faces = len(self.roof.roof_faces)
        if num_faces == 0:
            print("No roof faces found")
            return

        # Calculate subplot layout
        cols = min(3, num_faces)
        rows = (num_faces + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if num_faces == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i in range(num_faces):
            face = self.roof.roof_faces[i]
            projection_data = self._get_face_projection_data(face, grid_size)

            if projection_data is not None:
                self._plot_2d_projection_on_axis(projection_data, i, axes[i])
            else:
                axes[i].text(0.5, 0.5, f'Face {i}\n(Cannot process)',
                             ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'Face {i}')

        for i in range(num_faces, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()

    def _get_face_projection_data(self, face, grid_size):
        """
        Extract 2D projection data for a face (similar to process_face method)
        """
        face_verts = [self.roof.V[i].tolist() for i in face]
        normal = self.roof.compute_normal(face_verts)
        if normal is None:
            return None

        # Find edges at minimum Z level
        face_z = [v[2] for v in face_verts]
        min_z = min(face_z)
        edges = []
        n = len(face)
        for i in range(n):
            current = face[i]
            next_idx = face[(i + 1) % n]
            a = self.roof.V[current]
            b = self.roof.V[next_idx]
            edges.append((a, b))

        candidate_edges = [(a, b) for a, b in edges if a[2] == min_z and b[2] == min_z]
        if not candidate_edges:
            return None

        # Set up coordinate system
        A = np.array(candidate_edges[0][0])
        B = np.array(candidate_edges[0][1])
        u_vector = B - A
        u_norm = np.linalg.norm(u_vector)
        if u_norm < 1e-6:
            return None
        u_vector = u_vector / u_norm

        v_vector = np.cross(normal, u_vector)
        v_norm = np.linalg.norm(v_vector)
        if v_norm < 1e-6:
            return None
        v_vector = v_vector / v_norm

        # Project vertices to 2D
        uv_coords = []
        for idx in face:
            vertex = np.array(self.roof.V[idx])
            rel_vec = vertex - A
            u = np.dot(rel_vec, u_vector)
            v = np.dot(rel_vec, v_vector)
            uv_coords.append((u, v))

        u_coords = [u for u, v in uv_coords]
        v_coords = [v for u, v in uv_coords]
        u_min, u_max = min(u_coords), max(u_coords)
        v_min, v_max = min(v_coords), max(v_coords)

        # Generate grid squares
        squares = []
        all_squares = []  # Include invalid squares for visualization

        current_u = u_min
        while current_u < u_max:
            current_v = v_min
            while current_v < v_max:
                square_u_end = min(current_u + grid_size, u_max)
                square_v_end = min(current_v + grid_size, v_max)

                # Calculate actual dimensions
                u_length = square_u_end - current_u
                v_length = square_v_end - current_v
                min_dimension = grid_size * 0.3

                square_info = {
                    'bounds': (current_u, current_v, square_u_end, square_v_end),
                    'valid_size': u_length >= min_dimension and v_length >= min_dimension,
                    'inside': False
                }

                if square_info['valid_size']:
                    # Check if square is inside polygon
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
                        if not self.roof.point_in_polygon(u, v, uv_coords):
                            all_inside = False
                            break

                    if all_inside:
                        squares.append(square_info['bounds'])
                        square_info['inside'] = True

                all_squares.append(square_info)
                current_v += grid_size
            current_u += grid_size

        return {
            'uv_coords': uv_coords,
            'bounds': (u_min, u_max, v_min, v_max),
            'valid_squares': squares,
            'all_squares': all_squares,
            'grid_size': grid_size,
            'face_verts_3d': face_verts
        }

    def _plot_2d_projection(self, data, face_index):
        """
        Plot the 2D projection with grid
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        self._plot_2d_projection_on_axis(data, face_index, ax)
        plt.show()

    def _plot_2d_projection_on_axis(self, data, face_index, ax):
        """
        Plot the 2D projection on a given axis
        """
        uv_coords = data['uv_coords']
        u_min, u_max, v_min, v_max = data['bounds']
        valid_squares = data['valid_squares']
        all_squares = data['all_squares']
        grid_size = data['grid_size']

        polygon_x = [coord[0] for coord in uv_coords] + [uv_coords[0][0]]
        polygon_y = [coord[1] for coord in uv_coords] + [uv_coords[0][1]]
        ax.plot(polygon_x, polygon_y, 'k-', linewidth=2)

        poly = Polygon(uv_coords, alpha=0.2, facecolor='lightblue', edgecolor='black')
        ax.add_patch(poly)

        vertex_x = [coord[0] for coord in uv_coords]
        vertex_y = [coord[1] for coord in uv_coords]
        ax.scatter(vertex_x, vertex_y, c='red', s=50, zorder=5)

        for i, (x, y) in enumerate(uv_coords):
            ax.annotate(f'V{i}', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)

        for square_info in all_squares:
            start_u, start_v, end_u, end_v = square_info['bounds']
            width = end_u - start_u
            height = end_v - start_v

            if square_info['inside']:
                # Valid squares - green
                color = 'green'
                alpha = 0.3
            elif square_info['valid_size']:
                # Outside polygon but valid size - orange
                color = 'orange'
                alpha = 0.2
            else:
                # Too small - red
                color = 'red'
                alpha = 0.1

            rect = Rectangle((start_u, start_v), width, height,
                             linewidth=0.5, edgecolor='black',
                             facecolor=color, alpha=alpha)
            ax.add_patch(rect)

        ax.set_aspect('equal')
        ax.set_xlabel('U coordinate')
        ax.set_ylabel('V coordinate')
        ax.set_title(f'2D Projection of Face {face_index}\n'
                     f'Grid Size: {grid_size:.1f}, Valid Squares: {len(valid_squares)}', fontsize=10)
        ax.grid(True, alpha=0.3)

        u_range = u_max - u_min
        v_range = v_max - v_min
        padding = max(u_range, v_range) * 0.1
        ax.set_xlim(u_min - padding, u_max + padding)
        ax.set_ylim(v_min - padding, v_max + padding)


# Usage example function
def visualize_roof_projections(roof_solar_panel, face_index=None):
    """
    Convenience function to visualize roof face projections

    Args:
        roof_solar_panel: RoofSolarPanel instance
        face_index: Specific face to visualize (None for all faces)
    """
    visualizer = FaceProjectionVisualizer(roof_solar_panel)

    if face_index is not None:
        visualizer.visualize_face_projection(face_index)
    else:
        visualizer.visualize_all_faces()




class RoofSolarPanel:
    """
    Attributes:
        V (np.ndarray): Nx3 array of vertices (x,y,z coordinates)
        F (List[List[int]]): List of face vertex indices
        panel_dx (float): Panel x-dimension
        panel_dy (float): Panel y-dimension
        max_panels (int): Maximum number of panels allowed
        face_info (Dict): Stores calculated face properties

        b_scale_x (float): Scaling factor for building x-dimension
        b_scale_y (float): Scaling factor for building y-dimension
        b_scale_z (float): Scaling factor for building z-dimension

        p_scale_x (float): Scaling factor for panel x-dimension
        p_scale_y (float): Scaling factor for panel y-dimension

        panels (List[Dict]): Stores placed panel information
    """

    def __init__(self,
                 V: np.ndarray,
                 F: List[List[int]],
                 panel_dx: float,
                 panel_dy: float,
                 max_panels: int,
                 b_scale_x: float = 0.5,
                 b_scale_y: float = 0.5,
                 b_scale_z: float = 0.5,
                 p_scale_x: float = 1.0,
                 p_scale_y: float = 1.0,
                 grid_size: float = 15.0,
                 exclude_face_indices: List[int] = None):  # New parameter

        self.p_scale_x = p_scale_x
        self.p_scale_y = p_scale_y
        self.b_scale_x = b_scale_x
        self.b_scale_y = b_scale_y
        self.b_scale_z = b_scale_z
        self.grid_size = grid_size

        # Scale the vertices
        self.V = self._scale_vertices(V)
        self.F = F
        self.triangular_F = self.triangulate_all_faces()

        self.panel_dx = panel_dx * self.p_scale_x
        self.panel_dy = panel_dy * self.p_scale_y
        self.max_panels = max_panels

        self.face_info = {}
        self.panels = []

        # Store excluded face indices
        self.exclude_face_indices = exclude_face_indices if exclude_face_indices is not None else []

        # FIRST: Identify and filter roof faces
        original_roof_faces = read_polyshape_3d.identify_rooftops(self.V, self.F)
        self.roof_faces = []
        for face in original_roof_faces:
            try:
                idx = self.F.index(face)
                if idx not in self.exclude_face_indices:
                    self.roof_faces.append(face)
            except ValueError:
                continue

        # SECOND: Generate mesh objects AFTER filtering
        self.mesh_objects = self._generate_mesh()

        # THIRD: Process meshes AFTER creation
        self.flattened_meshes = []
        for face_mesh in self.mesh_objects:
            self.flattened_meshes.extend(face_mesh)

        self.triangular_meshes = []
        for mesh_idx, square in enumerate(self.flattened_meshes):
            tri1 = [square[0], square[1], square[2]]
            tri2 = [square[0], square[2], square[3]]
            self.triangular_meshes.append({'mesh_idx': mesh_idx, 'triangle': tri1})
            self.triangular_meshes.append({'mesh_idx': mesh_idx, 'triangle': tri2})

    def _generate_mesh(self):
        # Use self.roof_faces which already excludes the specified faces
        self.mesh_objects = [self.process_face(face, self.grid_size) for face in self.roof_faces]
        return self.mesh_objects
        """
        Initialize the roof model and planning parameters

        Args:
            V: Vertex coordinates array (Nx3)  -- in decimeters? dm
            F: List of face vertex indices  -- in decimeters? dm
            panel_dx: Panel length (x-dimension)
            panel_dy: Panel width (y-dimension)
            max_panels: Maximum number of panels to place
            b_scale_x, b_scale_y, b_scale_z: Scaling factors for building dimensions
            p_scale_x, p_scale_y: Scaling factors for panel dimensions
            grid_size: Size to create the mesh grid   -- in decimeters? dm

        self.p_scale_x = p_scale_x
        self.p_scale_y = p_scale_y
        self.b_scale_x = b_scale_x
        self.b_scale_y = b_scale_y
        self.b_scale_z = b_scale_z
        self.grid_size = grid_size

        # Scale the vertices
        self.V = self._scale_vertices(V)
        self.F = F
        self.triangular_F = self.triangulate_all_faces()
        #self.original_F = F  # Store original faces
        #self.F = self.triangulate_all_faces()  # Triangulate all faces

        self.panel_dx = panel_dx * self.p_scale_x
        self.panel_dy = panel_dy * self.p_scale_y
        self.max_panels = max_panels

        # Calculated properties
        self.face_info = {}
        self.panels = []

        # When instantiating the object, generate the mesh points
        self.mesh_objects = self._generate_mesh()

        # Store flattened mesh squares and generate triangular meshes
        self.flattened_meshes = []
        for face_mesh in self.mesh_objects:
            self.flattened_meshes.extend(face_mesh)

        self.triangular_meshes = []
        for mesh_idx, square in enumerate(self.flattened_meshes):
            # Split each square into two triangles
            tri1 = [square[0], square[1], square[2]]
            tri2 = [square[0], square[2], square[3]]
            self.triangular_meshes.append({'mesh_idx': mesh_idx, 'triangle': tri1})
            self.triangular_meshes.append({'mesh_idx': mesh_idx, 'triangle': tri2})

        # Store the roof faces
        self.roof_faces = read_polyshape_3d.identify_rooftops(self.V, self.F)         """


    """
    def _generate_mesh(self):
        roof_faces = read_polyshape_3d.identify_rooftops(self.V, self.F)
        self.mesh_objects = [self.process_face(face, self.grid_size) for face in roof_faces]
        return self.mesh_objects 
    """


    def plot_triangular_meshes(self):
        """
        Plots all triangular meshes generated from the rooftop grid squares.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        triangles = [tri_data['triangle'] for tri_data in self.triangular_meshes]

        mesh = Poly3DCollection(triangles, alpha=0.5, edgecolor='k', facecolor='cyan')
        ax.add_collection3d(mesh)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Triangular Meshes on Rooftop')
        ax.view_init(elev=20, azim=-45)
        plt.show()

    def triangulate_all_faces(self):
        triangulated = []
        for face in self.F:
            triangulated.extend(self.triangulate_face(face))
        return triangulated

    def triangulate_face(self, face):
        n = len(face)
        if n < 3:
            return []
        if n == 3:
            return [face]
        triangles = []
        for i in range(1, n - 1):
            triangles.append([face[0], face[i], face[i + 1]])
        return triangles

    def _scale_vertices(self, V: np.ndarray) -> np.ndarray:
        """
        Scale the vertices by the scaling factors.
        Args:
            V: Vertex coordinates array (Nx3)
        Returns:
            Scaled vertex coordinates array (Nx3)
        """
        #print(V)
        scaling_matrix = np.array([self.b_scale_x, self.b_scale_y, self.b_scale_z])
        #print(V * scaling_matrix)
        return V * scaling_matrix

    def display_building_and_rooftops(self):
        roof_faces = read_polyshape_3d.identify_rooftops(self.V, self.F)
        read_polyshape_3d.plot_building(self.V, self.F)
        read_polyshape_3d.plot_rooftops(self.V, roof_faces)

    def display_building_and_rooftops_triangulated(self):
        roof_faces = read_polyshape_3d.identify_rooftops(self.V, self.triangular_F)
        read_polyshape_3d.plot_building(self.V, self.triangular_F)

    def adjust_vertices_to_plane(self, vertices, faces):
        """
        Adjusts the vertices of a mesh so that the vertices of each face lie on the same plane.

        Args:
            vertices (np.ndarray): Nx3 array of vertex coordinates.
            faces (List[List[int]]): List of face vertex indices.

        Returns:
            np.ndarray: Adjusted vertex coordinates.

        # Step 1: Compute centroid
        # Step 2: Form the matrix A (3xN where each column is (p_i - centroid))
        # Step 3: Compute SVD of A
        # Step 4: The normal is the third column of U
        # Step 5: Project each vertex onto the plane
        """
        adjusted_verts = np.copy(vertices)

        for face in faces:
            indices = face
            points = adjusted_verts[indices]

            centroid = np.mean(points, axis=0)
            A = (points - centroid).T  # Transpose to get 3xN matrix
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            normal = U[:, 2]

            for idx in indices:
                point = adjusted_verts[idx]
                # Calculate the distance component along the normal
                distance = np.dot(point - centroid, normal)
                # Subtract this component from the original point to project onto the plane
                adjusted_point = point - distance * normal
                adjusted_verts[idx] = adjusted_point
                self.V = adjusted_verts

        return adjusted_verts

    def compute_normal(self, vertices):
        # Convert to numpy array
        points = np.array(vertices)
        centroid = np.mean(points, axis=0)
        shifted = points - centroid
        U, S, Vt = np.linalg.svd(shifted)
        normal = Vt[2]
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            return None
        return normal / norm

    def point_in_polygon(self, px, py, polygon, tol=1e-8):
        """
        Check if points reside inside the polygon
        """
        n = len(polygon)
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            if self.is_point_on_segment(px, py, x1, y1, x2, y2, tol):
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

    def is_point_on_segment(self, px, py, x1, y1, x2, y2, tol=1e-8):
        cross_product = (py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)
        if abs(cross_product) > tol:
            return False
        min_x = min(x1, x2) - tol
        max_x = max(x1, x2) + tol
        min_y = min(y1, y2) - tol
        max_y = max(y1, y2) + tol
        return (px >= min_x and px <= max_x) and (py >= min_y and py <= max_y)

    def process_face(self, face, grid_size = 15.0):
        face_verts = [self.V[i].tolist() for i in face]
        normal = self.compute_normal(face_verts)
        if normal is None:
            return []

        face_z = [v[2] for v in face_verts]
        min_z = min(face_z)
        edges = []
        n = len(face)
        for i in range(n):
            current = face[i]
            next_idx = face[(i + 1) % n]
            a = self.V[current]
            b = self.V[next_idx]
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
            vertex = np.array(self.V[idx])
            rel_vec = vertex - A
            u = np.dot(rel_vec, u_vector)
            v = np.dot(rel_vec, v_vector)
            uv_coords.append((u, v))

        u_coords = [u for u, v in uv_coords]
        v_coords = [v for u, v in uv_coords]
        u_min, u_max = min(u_coords), max(u_coords)
        v_min, v_max = min(v_coords), max(v_coords)

        """
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
                    if not self.point_in_polygon(u, v, uv_coords):
                        all_inside = False
                        break
                if all_inside:
                    squares.append((current_u, current_v, square_u_end, square_v_end))
                current_v += grid_size
            current_u += grid_size
        """

        squares = []
        current_u = u_min
        while current_u < u_max:
            current_v = v_min
            while current_v < v_max:
                square_u_end = min(current_u + grid_size, u_max)
                square_v_end = min(current_v + grid_size, v_max)

                u_length = square_u_end - current_u
                v_length = square_v_end - current_v
                min_dimension = grid_size * 0.3  # Adjust this threshold as needed

                # Skip squares that are too small in either dimension
                if u_length < min_dimension or v_length < min_dimension:
                    current_v += grid_size
                    continue  # Skip this square

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
                    if not self.point_in_polygon(u, v, uv_coords):
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
    """
    def _generate_mesh(self):
        roof_faces = read_polyshape_3d.identify_rooftops(self.V, self.F)
        self.mesh_objects = [self.process_face(face, self.grid_size) for face in roof_faces]
        return self.mesh_objects
    """

    def plot_rooftops_with_mesh_points(self):
        """
        Plots the rooftop structure along with the generated mesh grid on top.

        Arguments can be added/modified if necessary
        NO Args:
            verts (np.ndarray): Nx3 array of vertex coordinates.
            roof_faces (List[List[int]]): List of faces, where each face is a list of vertex indices.
            mesh_objects (List[List[List[float]]]): List of mesh squares for each face,
                                                    where each square is a list of 3D points.
        """
        roof_faces = self.roof_faces
        verts = self.V
        mesh_objects = self.mesh_objects

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c='grey', alpha=0.3)

        for face in roof_faces:
            poly = verts[face]
            ax.plot_trisurf(poly[:, 0], poly[:, 1], poly[:, 2], alpha=0.5, color='cyan', edgecolor='k')

        for mesh in mesh_objects:
            for square in mesh:
                square_points = np.array(square)
                ax.scatter(square_points[:, 0], square_points[:, 1], square_points[:, 2], c='red', s=2, alpha=0.7)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Rooftop with Mesh Points")
        plt.legend(loc='upper left', markerscale=2, fontsize=8)
        plt.show()


    def plot_rooftops_with_mesh_grid(self):
        """
        Plots the rooftop structure along with the generated mesh grid as quadrilaterals.

        NO Args:
            verts (np.ndarray): Nx3 array of vertex coordinates.
            roof_faces (List[List[int]]): List of faces, where each face is a list of vertex indices.
            mesh_objects (List[List[List[float]]]): List of mesh squares for each face,
                                                    where each square is a list of 3D points.
        """
        roof_faces = self.roof_faces
        verts = self.V
        mesh_objects = self.mesh_objects

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c='grey', alpha=0.3, label='Vertices')
        for face in roof_faces:
            poly = verts[face]
            #ax.plot_trisurf(poly[:, 0], poly[:, 1], poly[:, 2], alpha=0.5, color='cyan', edgecolor='k')

        for mesh in mesh_objects:
            for square in mesh:
                square_points = np.array(square)
                x = square_points[:, 0]
                y = square_points[:, 1]
                z = square_points[:, 2]
                ax.plot_trisurf(x, y, z, color='blue', alpha=0.7, edgecolor='black')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Rooftop with Mesh Grid")
        plt.legend(loc='upper left', markerscale=2, fontsize=8)
        plt.show()


    def plot_building_with_tri_mesh_grid(self):
        """
        Plots the entire building structure along with the generated mesh grid as quadrilaterals.
        Shows triangulated lines only on mesh grid, not on roof segments.
        """
        verts = self.V
        mesh_objects = self.mesh_objects
        faces = self.F  # All building faces, not just roof faces

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c='grey', alpha=0.3, s=10, label='Vertices')

        for face in faces:
            polygon = [verts[i] for i in face]
            poly = Poly3DCollection([polygon], alpha=0.3, linewidths=0.5)
            # Use light gray color for building to make it less prominent
            poly.set_facecolor('cyan')
            poly.set_edgecolor('k')
            ax.add_collection3d(poly)

        for mesh in mesh_objects:
            for square in mesh:
                square_points = np.array(square)
                x = square_points[:, 0]
                y = square_points[:, 1]
                z = square_points[:, 2]
                ax.plot_trisurf(x, y, z, color='blue', alpha=0.5,
                                edgecolor='black')

        min_vals = verts.min(axis=0)
        max_vals = verts.max(axis=0)
        ax.set_xlim(min_vals[0], max_vals[0])
        ax.set_ylim(min_vals[1], max_vals[1])
        ax.set_zlim(min_vals[2], max_vals[2])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title("Building with Prominent Triangulated Mesh Grid")
        ax.view_init(elev=60, azim=-30)

        plt.legend(loc='upper left', markerscale=2, fontsize=8)
        plt.show()

    def plot_building_with_mesh_grid(self):
        """
        Plots the 3D building with the generated mesh grid on top.

        NO Args:
            verts (np.ndarray): Nx3 array of vertex coordinates.
            faces (List[List[int]]): List of faces, where each face is a list of vertex indices.
            mesh_objects (List[List[List[float]]]): List of mesh squares for each face,
                                                    where each square is a list of 3D points.
        """
        verts = self.V
        mesh_objects = self.mesh_objects
        faces = self.roof_faces

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')

        for face in faces:
            polygon = [verts[i] for i in face]
            poly = Poly3DCollection([polygon], alpha=0.3, edgecolor='k', linewidths=1)
            poly.set_facecolor('cyan')  # Changed from random colors to cyan
            poly.set_edgecolor('k')
            ax.add_collection3d(poly)

        for mesh in mesh_objects:
            for square in mesh:
                square_points = np.array(square)
                poly = Poly3DCollection([square_points], alpha=0.5, edgecolor='black', linewidths=0.5)
                poly.set_facecolor('blue')
                poly.set_edgecolor('black')
                ax.add_collection3d(poly)

        min_vals = verts.min(axis=0)
        max_vals = verts.max(axis=0)
        ax.set_xlim(min_vals[0], max_vals[0])
        ax.set_ylim(min_vals[1], max_vals[1])
        ax.set_zlim(min_vals[2], max_vals[2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.view_init(elev=60, azim=-30)
        plt.title('Building Model with Mesh Grid')
        plt.show()


    def get_ground_centroid(self):
        """
        Calculate the centroid of the ground floor vertices.
        - vertices: List of (x, y, z) tuples
        Returns: (centroid_x, centroid_y, ground_z) tuple
        """
        ground_z = min(v[2] for v in self.V)
        ground_verts = [v for v in self.V if v[2] == ground_z]

        if not ground_verts:
            raise ValueError("No ground vertices found")

        centroid_x = sum(v[0] for v in ground_verts) / len(ground_verts)
        centroid_y = sum(v[1] for v in ground_verts) / len(ground_verts)

        return (centroid_x, centroid_y, ground_z)


if __name__ == "__main__":
    # Load vertices and faces from a .polyshape file
    verts, faces = read_polyshape_3d.read_polyshape(
        "C:/Users/Sharon/Desktop/SGA21_roofOptimization-main/SGA21_roofOptimization-main/RoofGraphDataset/res_building/BK39_500_014034_0007.polyshape"
    )
    #BK39_500_013033_0004
    #BK39_500_012023_0006.polyshape

    roof = RoofSolarPanel(
        V= verts,
        F= faces,
        panel_dx=2.0,
        panel_dy=1.0,
        max_panels=10,
        b_scale_x=0.05,
        b_scale_y=0.05,
        b_scale_z=0.05,
        grid_size= 1.0,
        exclude_face_indices=[]   # 2 for test 2   #9 for test
    )
    #print(roof.V)
    #print(roof.F)

    # Coplanarity
    #roof.adjust_vertices_to_plane(roof.V, roof.F)

    # Display the building and rooftop seperately
    # roof.display_building_and_rooftops()
    # roof.display_building_and_rooftops_triangulated()
    # print(roof.triangular_F)

    # The mesh object of the rooftop
    # print(roof.mesh_objects)

    # display the rooftop with mesh points
    # roof.plot_rooftops_with_mesh_points()

    # display the rooftop with mesh grid
    roof.plot_building_with_tri_mesh_grid()

    # display the building with mesh grid
    # roof.plot_building_with_mesh_grid()

    #print(roof.mesh_objects)
    #print(roof.roof_faces)

    # get the centroid of the ground floor
    # print(roof.get_ground_centroid())
    # roof.plot_triangular_meshes()
    # print(roof.triangular_meshes)
    # print(roof.get_ground_centroid())

    # Then add these lines to visualize projections:
    # visualizer = FaceProjectionVisualizer(roof)

    # Visualize specific face (e.g., first roof face)
    # visualizer.visualize_face_projection(0)
    # visualizer.visualize_all_faces()