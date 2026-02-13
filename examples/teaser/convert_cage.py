f = open("./meshes/cage_0080.obj")



tet6_edges = [
  [0, 1], [1, 3], [3, 7], [7, 0],
  [0, 3], [3, 2], [2, 7], [7, 0],
  [0, 2], [2, 6], [6, 7], [7, 0],
  [0, 6], [6, 4], [4, 7], [7, 0],
  [0, 4], [4, 5], [5, 7], [7, 0],
  [0, 5], [5, 1], [1, 7], [7, 0],
]

face_defs = [
  [0, 2, 6, 4],  # x-min
  [1, 3, 7, 5],  # x-max
  [0, 1, 5, 4],  # y-min
  [2, 3, 7, 6],  # y-max
  [0, 1, 3, 2],  # z-min
  [4, 5, 7, 6],  # z-max
]

face_edges = []
for face in face_defs:
  for i in range(4):
    face_edges.append([face[i], face[(i + 1) % 4]])

edge_indices = []
for i in range(len(tet6_edges)):
  if tet6_edges[i] in face_edges:
    print(i)
    edge_indices.append(i)
  # check reverse
  tmp = [tet6_edges[i][1], tet6_edges[i][0]]
  if tmp in face_edges:
    print(i)
    edge_indices.append(i)
edge_indices = list(set(edge_indices))


lines = []

cage_lines = 0
for line in f:
  if line.startswith("l"):
    if cage_lines % 24 in edge_indices:
      lines.append(line)
    cage_lines += 1
  else:
    lines.append(line)
f.close()

f = open("cage_0080.obj", "w")
f.writelines(lines)
f.close()
