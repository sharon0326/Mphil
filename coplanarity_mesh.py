from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import read_polyshape_3d
import numpy as np
from typing import List, Dict

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
        return self.mesh_objects """


    def plot_triangular_meshes(self):
        """
        Plots all triangular meshes generated from the rooftop grid squares.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Extract all triangles from triangular_meshes
        triangles = [tri_data['triangle'] for tri_data in self.triangular_meshes]

        # Create a Poly3DCollection and add to plot
        mesh = Poly3DCollection(triangles, alpha=0.5, edgecolor='k', facecolor='cyan')
        ax.add_collection3d(mesh)

        # Set labels and viewing angle
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
        # Identify rooftops from the faces
        roof_faces = read_polyshape_3d.identify_rooftops(self.V, self.F)
        # Plot the building
        read_polyshape_3d.plot_building(self.V, self.F)
        # Plot the rooftops
        read_polyshape_3d.plot_rooftops(self.V, roof_faces)

    def display_building_and_rooftops_triangulated(self):
        # Identify rooftops from the faces
        roof_faces = read_polyshape_3d.identify_rooftops(self.V, self.triangular_F)
        # Plot the building
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

                # Calculate actual dimensions of the square
                u_length = square_u_end - current_u
                v_length = square_v_end - current_v
                # Define minimum allowed dimension (e.g., 10% of grid_size)
                min_dimension = grid_size * 0.3  # Adjust this threshold as needed

                # Skip squares that are too small in either dimension
                if u_length < min_dimension or v_length < min_dimension:
                    current_v += grid_size
                    continue  # Skip this square

                # Proceed with checking if points are inside the polygon
                corners = [
                    (current_u, current_v),
                    (square_u_end, current_v),
                    (square_u_end, square_v_end),
                    (current_u, square_v_end)
                ]
                # ... rest of the code to check points ...

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

            # ... rest of the code ...


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
        # can be added as arguments to the function
        roof_faces = self.roof_faces
        verts = self.V
        mesh_objects = self.mesh_objects

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c='grey', alpha=0.3)

        # Plot rooftop faces
        for face in roof_faces:
            poly = verts[face]
            ax.plot_trisurf(poly[:, 0], poly[:, 1], poly[:, 2], alpha=0.5, color='cyan', edgecolor='k')

        # Plot mesh grid points
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
        # can be added as arguments to the function
        roof_faces = self.roof_faces
        verts = self.V
        mesh_objects = self.mesh_objects

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot all vertices and rooftop faces
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c='grey', alpha=0.3, label='Vertices')
        for face in roof_faces:
            poly = verts[face]
            ax.plot_trisurf(poly[:, 0], poly[:, 1], poly[:, 2], alpha=0.5, color='cyan', edgecolor='k')

        # Plot the mesh grid as quadrilaterals
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

    def plot_building_with_mesh_grid(self):
        """
        Plots the 3D building with the generated mesh grid on top.

        NO Args:
            verts (np.ndarray): Nx3 array of vertex coordinates.
            faces (List[List[int]]): List of faces, where each face is a list of vertex indices.
            mesh_objects (List[List[List[float]]]): List of mesh squares for each face,
                                                    where each square is a list of 3D points.
        """
        # can be added as arguments to the function
        verts = self.V
        mesh_objects = self.mesh_objects
        faces = self.F

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot each face of the building
        for face in faces:
            polygon = [verts[i] for i in face]
            poly = Poly3DCollection([polygon], alpha=0.5, edgecolor='k', linewidths=1)
            poly.set_facecolor(np.random.rand(3, ))
            ax.add_collection3d(poly)

        # Plot the mesh grid as quadrilaterals
        for mesh in mesh_objects:
            for square in mesh:
                square_points = np.array(square)
                poly = Poly3DCollection([square_points], alpha=0.8, edgecolor='blue', linewidths=0.5)
                poly.set_facecolor('blue')
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
        # Find ground level (lowest Z)
        ground_z = min(v[2] for v in self.V)
        ground_verts = [v for v in self.V if v[2] == ground_z]

        if not ground_verts:
            raise ValueError("No ground vertices found")

        # Calculate X/Y centroid coordinates
        centroid_x = sum(v[0] for v in ground_verts) / len(ground_verts)
        centroid_y = sum(v[1] for v in ground_verts) / len(ground_verts)

        return (centroid_x, centroid_y, ground_z)


if __name__ == "__main__":
    # Load vertices and faces from a .polyshape file
    verts, faces = read_polyshape_3d.read_polyshape(
        "C:/Users/Sharon/Desktop/SGA21_roofOptimization-main/SGA21_roofOptimization-main/RoofGraphDataset/res_building/test2.txt"
    )

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
        exclude_face_indices=[2]
    )
    #print(roof.V)
    #print(roof.F)

    # Coplanarity
    #roof.adjust_vertices_to_plane(roof.V, roof.F)

    # Display the building and rooftop seperately
    roof.display_building_and_rooftops()
    roof.display_building_and_rooftops_triangulated()
    # print(roof.triangular_F)

    # The mesh object of the rooftop
    # print(roof.mesh_objects)

    # display the rooftop with mesh points
    # roof.plot_rooftops_with_mesh_points()

    # display the rooftop with mesh grid
    roof.plot_rooftops_with_mesh_grid()

    # display the building with mesh grid
    #roof.plot_building_with_mesh_grid()

    #print(roof.mesh_objects)
    #print(roof.roof_faces)

    # get the centroid of the ground floor
    # print(roof.get_ground_centroid())
    roof.plot_triangular_meshes()
    #print(roof.triangular_meshes)
    #print(roof.get_ground_centroid())