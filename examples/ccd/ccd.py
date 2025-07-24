from __future__ import annotations
# from ast import Str
from yasps.attribute import attribute
from typing import List
import pycuda.gpuarray as gpuarray
import ctypes
import numpy as np
from yasps.helper import timed
import os
import pycuda.driver as cuda
import subprocess


class CCD:
  def __init__(self, num_vertices: int, max_cd_pairs: int = 100000, max_ccd_pairs: int = 100000):
    double3_p = ctypes.c_void_p
    uint2_p = ctypes.c_void_p
    int4_p = ctypes.c_void_p
    uint32_p = ctypes.c_void_p
    int_p = ctypes.c_void_p
    if not os.path.exists("libmlbvh.so"):
      # we first compile the code
      compile_cmds = [
        [
          "nvcc", "-std=c++17", "-O3", "-Xcompiler", "-fPIC",
          "-I/usr/include/eigen", "-I.",
          "-gencode", "arch=compute_86,code=sm_86",
          "--relocatable-device-code=true",
          "-c", "mlbvh.cu", "-o", "mlbvh.o"
        ],
        [
          "nvcc", "-std=c++17", "-O3", "-Xcompiler", "-fPIC",
          "-I/usr/include/eigen", "-I.",
          "-gencode", "arch=compute_86,code=sm_86",
          "--relocatable-device-code=true",
          "-c", "gpu_eigen_libs.cu", "-o", "gpu_eigen_libs.o"
        ],
        [
          "nvcc", "-std=c++17", "-O3", "-Xcompiler", "-fPIC",
          "-gencode", "arch=compute_86,code=sm_86",
          "--relocatable-device-code=true",
          "mlbvh.o", "gpu_eigen_libs.o",
          "-o", "libmlbvh.so", "--shared"
        ]
      ]

      for cmd in compile_cmds:
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    # we are probably certain that the library is compiled
    # first get the file
    lib_path = os.path.join(os.path.dirname(__file__), "libmlbvh.so")
    self.__mlbvh = ctypes.CDLL(lib_path)
    self.__mlbvh.create_lbvh_f.restype = ctypes.c_void_p
    self.__mlbvh.create_lbvh_e.restype = ctypes.c_void_p
    self.__bvh_f = self.__mlbvh.create_lbvh_f()
    self.__bvh_e = self.__mlbvh.create_lbvh_e()


    self.__lbvh_f_init = self.__mlbvh.lbvh_f_init
    self.__lbvh_e_init = self.__mlbvh.lbvh_e_init
    self.__lbvh_f_init.argtypes = [
      ctypes.c_void_p, # the object
      ctypes.c_void_p, # btype, gpu array pointer to int
      ctypes.c_void_p, # vertices, gpu array pointer to double3
      ctypes.c_void_p, # faces, gpu array pointer to int3
      ctypes.c_void_p, # surface vertices, gpu array pointer to uint32
      ctypes.c_void_p, # collision pairs, gpu array pointer to int4
      ctypes.c_void_p, # continuous collision pairs, gpu array pointer to int4
      ctypes.c_void_p, # mcp num
      ctypes.c_void_p, # mat index, useless for us but we will need to allocate it
      ctypes.c_int, # number of faces
      ctypes.c_int, # number of vertices
    ]
    self.__lbvh_e_init.argtypes = [
      ctypes.c_void_p, # the object
      ctypes.c_void_p, # btype, gpu array pointer to int
      ctypes.c_void_p, # vertices, gpu array pointer to double3
      ctypes.c_void_p, # rest vertices, gpu array pointer to double3
      ctypes.c_void_p, # edges, gpu array pointer to int3
      ctypes.c_void_p, # collision pairs, gpu array pointer to int4
      ctypes.c_void_p, # continuous collision pairs, gpu array pointer to int4
      ctypes.c_void_p, # mcp num
      ctypes.c_void_p, # mat index, useless for us but we will need to allocate it
      ctypes.c_uint32, # number of edges
      ctypes.c_uint32, # number of vertices
    ]

    self.__lbvh_e_construct = self.__mlbvh.lbvh_e_construct
    self.__lbvh_e_construct.argtypes = [
      ctypes.c_void_p, # the object
      ctypes.c_void_p # positions of the vertices
    ]

    self.__lbvh_f_construct = self.__mlbvh.lbvh_f_construct
    self.__lbvh_f_construct.argtypes = [
      ctypes.c_void_p, # the object
      ctypes.c_void_p # positions of the vertices
    ]

    self.__lbvh_f_construct_full_ccd = self.__mlbvh.lbvh_f_construct_full_ccd
    self.__lbvh_f_construct_full_ccd.argtypes = [
      ctypes.c_void_p, # the object
      ctypes.c_void_p, # moving directions
      ctypes.POINTER(ctypes.c_double)
    ]

    self.__collision_pairs = gpuarray.to_gpu(np.zeros((max_cd_pairs, 4), dtype=np.int32))
    assert self.__collision_pairs.ptr % 16 == 0, "Device memory is not 16-byte aligned!"
    self.__collision_pairs_ccd = gpuarray.to_gpu(np.zeros((max_ccd_pairs, 4), dtype=np.int32))
    assert self.__collision_pairs_ccd.ptr % 16 == 0, "Device memory is not 16-byte aligned!"
    self.__mat_index = gpuarray.to_gpu(np.zeros(max_cd_pairs, dtype=np.int32))
    self.__cp_num = gpuarray.to_gpu(np.zeros(5, dtype=np.int32)) # for some reason this is 5 in GIPC and we will keep it this way

    self.__num_vertices = num_vertices
    self.__btypes = gpuarray.to_gpu(np.zeros(num_vertices, dtype=np.int32)) # initialize empty array for btypes


  # def init_faces(self)
  def __to_void_p(self, x: gpuarray.GPUArray):
    if x is None or x.size == 0:
      # Return a NULL pointer if array is empty
      return ctypes.c_void_p(None)
    assert x.gpudata is not None
    return ctypes.c_void_p(int(x.gpudata))

  def init_edges(self,
    vertices: gpuarray.GPUArray, # list of double
    vertices_rest: gpuarray.GPUArray, # list of double
    edges, # for this we don't inpose a type, but we need it to be convertible to uint2
    edge_num: int, # number of edges
  ):
    self.__lbvh_e_init(
      self.__bvh_e,
      self.__to_void_p(self.__btypes),
      self.__to_void_p(vertices),
      self.__to_void_p(vertices_rest),
      self.__to_void_p(edges),
      self.__to_void_p(self.__collision_pairs),
      self.__to_void_p(self.__collision_pairs_ccd),
      self.__to_void_p(self.__cp_num),
      self.__to_void_p(self.__mat_index),
      ctypes.c_uint32(edge_num),
      ctypes.c_uint32(self.__num_vertices)
    )
  def construct_edges(self, vertices: gpuarray.GPUArray):
    self.__lbvh_e_construct(
      self.__bvh_e,
      self.__to_void_p(vertices)
    )

  def construct_full_ccd_edges(self, moving_directions: gpuarray.GPUArray, time_step: float):
    """
    Constructs the full CCD edges with the given moving directions and time step.
    """
    assert moving_directions.dtype == np.float64, "Moving directions must be of type float64"
    assert moving_directions.shape[0] == self.__num_vertices, "Moving directions must have the same number of elements as vertices"
    c_time_step = ctypes.c_double(time_step)
    time_step_p = ctypes.byref(c_time_step)
    self.__lbvh_f_construct_full_ccd(
      self.__bvh_f,
      self.__to_void_p(moving_directions),
      time_step_p
    )




x = CCD(10)
points = gpuarray.to_gpu(np.random.rand(10, 3).astype(np.float64))
points_rest = gpuarray.to_gpu(np.random.rand(10, 3).astype(np.float64))
edges = gpuarray.to_gpu(np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 0]], dtype=np.int32))
x.init_edges(points, points_rest, edges, 10)
x.construct_edges(points)
