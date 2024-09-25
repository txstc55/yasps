from yasps import scene
from yasps import codeGenerator
import numpy as np
# set seed
np.random.seed(0)
scene0 = scene("scene0") # create the scene
m1 = scene0.addMesh("mesh1") # add the mesh
box_vertices = m1.addPrimitive("box_vertices", 8) # add a primitive box vertex as control points
box_vertices.addAttribute("position", rows = 1, cols = 3) # box vertices have 3D positions
positions = scene0.mesh1.box_vertices["position"] # update the values
cube_positions_values = np.array([[-1, 1, 1], [1, 1, 1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, -1, -1], [-1, -1, -1]], dtype = np.float64)
positions.updateValue(cube_positions_values)
m1.addPrimitive("vertices", 6) # d8
m1.vertices.addConnectivity("box_to_vertex", m1.box_vertices, np.array([[0, 1, 2, 3], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [0, 4, 7, 3], [4, 5, 6, 7]]), 4) # define the relationship
m1.vertices.addAttribute("position", through = m1.vertices.box_to_vertex) # extract position from box vertices
m1.vertices.addAttribute("box_vertex_weights", rows = 1, cols = 4) # add attribute weight

# generate random weights
weights = np.array([[0.125, 0.375, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]])
def generate_random_weights(num_elements):
    random_weights = np.random.rand(num_elements)  # Generate random values
    return random_weights / random_weights.sum()   # Normalize to sum to 1

# Apply the random weight generation to each subarray
for i in range(weights.shape[0]):
    weights[i] = generate_random_weights(weights.shape[1])

m1.vertices["box_vertex_weights"].updateValue(weights) # add the values
# here we compute the weighted position
# weighted_position = m1.vertices["position"].row(0) * m1.vertices["box_vertex_weights"][0] + m1.vertices["position"].row(1) * m1.vertices["box_vertex_weights"][1] + m1.vertices["position"].row(2) * m1.vertices["box_vertex_weights"][2] + m1.vertices["position"].row(3) * m1.vertices["box_vertex_weights"][3]
#
weighted_position = m1.vertices["box_vertex_weights"] * m1.vertices["position"] # define the weighted positions
m1.vertices.addAttribute("weighted_position", computed_attribute = weighted_position) # add the attribute
attr = m1.vertices["weighted_position"]
attr.compute()


# visualize
import pyvista as pv
triangles = np.array([
    [0, 1, 2],
    [0, 2, 3],
    [0, 3, 4],
    [0, 4, 1],
    [1, 5, 2],
    [2, 5, 3],
    [3, 5, 4],
    [4, 5, 1]
  ]
)



cells = np.hstack([np.full((triangles.shape[0], 1), 3), triangles])
mesh = pv.PolyData(attr.value.get().reshape((6, 3)), cells)
# Create a PyVista plotter object
plotter = pv.Plotter()
actor = plotter.add_mesh(mesh, color="lightblue", show_edges=True)


# Create a PyVista PolyData object with only points (no faces)
cube = pv.PolyData(cube_positions_values)
plotter.add_mesh(cube, color='black', style='points', point_size=10)

# Bind the 'U' key to the update function
def key_callback_event(key):
  if key == "u":  # When 'U' key is pressed, update the mesh
    # Randomly choose a single index in the flattened array
    rows, cols = cube_positions_values.shape
    flat_idx = np.random.choice(rows * cols)
    # Convert the flat index to a 2D index
    row_idx, col_idx = divmod(flat_idx, cols)
    # Modify the randomly selected value
    if cube_positions_values[row_idx, col_idx] > 0:
        cube_positions_values[row_idx, col_idx] += 0.1
    else:
      cube_positions_values[row_idx, col_idx] -= 0.1

    # Update the attribute value
    positions.updateValue(cube_positions_values)

    attr.compute()
    mesh.points = attr.value.get().reshape((6, 3))
    cube.points = cube_positions_values

    # Trigger a render update
    plotter.render()

# Bind the 'U' key to the update function
plotter.add_key_event("u", lambda: key_callback_event("u"))

# Start the interactive plot
plotter.show()
