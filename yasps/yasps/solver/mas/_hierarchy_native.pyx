# cython: language_level=3, boundscheck=False, wraparound=False
"""Native loops for validated internal MAS topology arrays.

Public shape/range/permutation validation belongs to the Python callers.
No floating-point values or solver tolerances are involved here.
"""
from libc.stdint cimport int64_t, uint8_t
import numpy as np


cdef int64_t root(int64_t[::1] parent, int64_t node) noexcept nogil:
  while parent[node] != node:
    parent[node] = parent[parent[node]]
    node = parent[node]
  return node


def collapse(const int64_t[::1] xadj, const int64_t[::1] adjacency, const int64_t[::1] nodes, const int64_t[::1] offsets, const int64_t[::1] dimensions, const uint8_t[::1] compatible_types):
  cdef int64_t count = dimensions.shape[0]
  cdef int64_t[::1] parent = np.arange(count, dtype=np.int64)
  cdef int64_t[::1] owner = np.empty(count, dtype=np.int64)
  cdef int64_t domain, k, i, j, a, b
  cdef bint check_type = compatible_types.shape[0] != 0
  with nogil:
    for domain in range(offsets.shape[0] - 1):
      for k in range(offsets[domain], offsets[domain + 1]):
        owner[nodes[k]] = domain
    # Sorted CSR visits the same canonical undirected edge order as before.
    for i in range(count):
      for k in range(xadj[i], xadj[i + 1]):
        j = adjacency[k]
        if j <= i or owner[i] != owner[j] or dimensions[i] != dimensions[j]:
          continue
        if check_type and not compatible_types[k]:
          continue
        a, b = root(parent, i), root(parent, j)
        if a < b:
          parent[b] = a
        elif b < a:
          parent[a] = b
    for i in range(count):
      parent[i] = root(parent, i)
  # Minimum-root union means sorted roots equal first-occurrence numbering.
  return np.unique(np.asarray(parent), return_inverse=True)


def pack(const int64_t[::1] nodes, const int64_t[::1] offsets, const int64_t[::1] dimensions, int64_t capacity):
  cdef list values = np.asarray(nodes).tolist()
  cdef list domains = []
  cdef int64_t group, first, last, k, total, dim
  for group in range(offsets.shape[0] - 1):
    first, last = offsets[group], offsets[group + 1]
    total = 0
    for k in range(first, last):
      dim = dimensions[nodes[k]]
      if first < k and total + dim > capacity:
        domains.append(values[first:k])
        first, total = k, 0
      total += dim
      if dim > capacity:
        domains.append(values[first:k + 1])
        first, total = k + 1, 0
    if first < last:
      domains.append(values[first:last])
  return domains


def component_order(const int64_t[::1] xadj, const int64_t[::1] adjacency, const int64_t[::1] labels):
  """BFS within each METIS label, ordered by minimum component vertex."""
  cdef int64_t count = labels.shape[0]
  order_array = np.empty(count, dtype=np.int64)
  cdef int64_t[::1] order = order_array
  cdef int64_t[::1] starts = np.empty(count + 1, dtype=np.int64)
  cdef uint8_t[::1] visited = np.zeros(count, dtype=np.uint8)
  cdef int64_t i, j, k, node, head, tail = 0, groups = 0
  with nogil:
    for i in range(count):
      if visited[i]:
        continue
      starts[groups] = tail
      groups += 1
      head = tail
      order[tail] = i
      tail += 1
      visited[i] = 1
      while head < tail:
        node = order[head]
        head += 1
        for k in range(xadj[node], xadj[node + 1]):
          j = adjacency[k]
          if not visited[j] and labels[j] == labels[node]:
            visited[j] = 1
            order[tail] = j
            tail += 1
    starts[groups] = tail
  cdef list values = order_array.tolist()
  cdef list output = []
  for i in range(groups):
    output.append(tuple(values[starts[i]:starts[i + 1]]))
  return order_array, tuple(output)
