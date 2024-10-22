# manually test out the energy
p0 = vertex_positions_value[0, :]
p1 = vertex_positions_value[1, :]
p2 = vertex_positions_value[2, :]
p3 = vertex_positions_value[3, :]
p4 = vertex_positions_value[4, :]
def repulsive_energy_np(p0, p1, p2, p3, repulsive_weight, alpha, beta):
  T01 = (p1 - p0) / math.sqrt((p1 - p0).dot(p1 - p0))
  T23 = (p3 - p2) / math.sqrt((p3 - p2).dot(p3 - p2))
  r = math.pow(T01.dot(p0 - p2), alpha) / math.pow(np.linalg.norm(p0 - p2), beta)
  r += math.pow(T01.dot(p0 - p3), alpha) / math.pow(np.linalg.norm(p0 - p3), beta)
  r += math.pow(T01.dot(p1 - p2), alpha) / math.pow(np.linalg.norm(p1 - p2), beta)
  r += math.pow(T01.dot(p1 - p3), alpha) / math.pow(np.linalg.norm(p1 - p3), beta)
  r += math.pow(T23.dot(p2 - p0), alpha) / math.pow(np.linalg.norm(p2 - p0), beta)
  r += math.pow(T23.dot(p2 - p1), alpha) / math.pow(np.linalg.norm(p2 - p1), beta)
  r += math.pow(T23.dot(p3 - p0), alpha) / math.pow(np.linalg.norm(p3 - p0), beta)
  r += math.pow(T23.dot(p3 - p1), alpha) / math.pow(np.linalg.norm(p3 - p1), beta)
  return r * repulsive_weight / 4.0
