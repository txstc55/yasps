import igl
import numpy as np

V, F = igl.read_triangle_mesh("../data/bunny_small.obj")
Vs = igl.map_vertices_to_sphere(V, F)
igl.write_triangle_mesh("bunny_sphere.obj", Vs, F)
