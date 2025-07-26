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

    self.__lbvh_f_construct = self.__mlbvh.lbvh_f_construct
    self.__lbvh_f_construct.argtypes = [
      ctypes.c_void_p, # the object
      ctypes.c_void_p # positions of the vertices
    ]

    self.__lbvh_e_construct = self.__mlbvh.lbvh_e_construct
    self.__lbvh_e_construct.argtypes = [
      ctypes.c_void_p, # the object
      ctypes.c_void_p # positions of the vertices
    ]



    self.__lbvh_f_construct_full_ccd = self.__mlbvh.lbvh_f_construct_full_ccd
    self.__lbvh_f_construct_full_ccd.argtypes = [
      ctypes.c_void_p, # the object
      ctypes.c_void_p, # positions of the vertices
      ctypes.c_void_p, # moving directions
      ctypes.POINTER(ctypes.c_double)
    ]

    self.__lbvh_e_construct_full_ccd = self.__mlbvh.lbvh_e_construct_full_ccd
    self.__lbvh_e_construct_full_ccd.argtypes = [
      ctypes.c_void_p, # the object
      ctypes.c_void_p, # positions of the vertices
      ctypes.c_void_p, # moving directions
      ctypes.POINTER(ctypes.c_double)
    ]

    self.__lbvh_f_self_collision_detect = self.__mlbvh.lbvh_f_self_collision_detect
    self.__lbvh_f_self_collision_detect.argtypes = [
      ctypes.c_void_p, # the object
      ctypes.c_double, # dhat
    ]
    self.__lbvh_e_self_collision_detect = self.__mlbvh.lbvh_e_self_collision_detect
    self.__lbvh_e_self_collision_detect.argtypes = [
      ctypes.c_void_p, # the object
      ctypes.c_double, # dhat
    ]

    self.__lbvh_f_self_collision_full_detect = self.__mlbvh.lbvh_f_self_collision_full_detect
    self.__lbvh_f_self_collision_full_detect.argtypes = [
      ctypes.c_void_p, # the object
      ctypes.c_double, #dhat
      ctypes.c_void_p, # moving directions
      ctypes.POINTER(ctypes.c_double) # alpha
    ]
    self.__lbvh_e_self_collision_full_detect = self.__mlbvh.lbvh_e_self_collision_full_detect
    self.__lbvh_e_self_collision_full_detect.argtypes = [
      ctypes.c_void_p, # the object
      ctypes.c_double, #dhat
      ctypes.c_void_p, # moving directions
      ctypes.POINTER(ctypes.c_double) # alpha
    ]

    self.__collision_pairs = gpuarray.to_gpu(np.zeros((max_cd_pairs, 4), dtype=np.int32))
    assert self.__collision_pairs.ptr % 16 == 0, "Device memory is not 16-byte aligned!"
    self.__collision_pairs_ccd = gpuarray.to_gpu(np.zeros((max_ccd_pairs, 4), dtype=np.int32))
    assert self.__collision_pairs_ccd.ptr % 16 == 0, "Device memory is not 16-byte aligned!"
    self.__mat_index = gpuarray.to_gpu(np.zeros(max_cd_pairs, dtype=np.int32))
    self.__cp_num = gpuarray.to_gpu(np.zeros(5, dtype=np.int32)) # for some reason this is 5 in GIPC and we will keep it this way

    self.__num_vertices = num_vertices
    self.__btypes = gpuarray.to_gpu(np.zeros(num_vertices, dtype=np.int32)) # initialize empty array for btypes
    self.__pp = gpuarray.to_gpu(np.zeros((max_ccd_pairs, 2), dtype=np.uint32))
    self.__pe = gpuarray.to_gpu(np.zeros((max_ccd_pairs, 3), dtype=np.uint32))
    self.__pt = gpuarray.to_gpu(np.zeros((max_ccd_pairs, 4), dtype=np.uint32))
    self.__ee = gpuarray.to_gpu(np.zeros((max_ccd_pairs, 4), dtype=np.uint32))
    self.__separated_counts = gpuarray.to_gpu(np.zeros(3, dtype=np.uint32))

    self.__lbvh_f_separate_cases = self.__mlbvh.lbvh_f_separate_cases
    self.__lbvh_f_separate_cases.argtypes = [
      ctypes.c_void_p, # the object
      ctypes.c_void_p, # pp
      ctypes.c_void_p, # pe
      ctypes.c_void_p, # pt
      ctypes.c_void_p, # separated counts
    ]
    self.__lbvh_e_separate_cases = self.__mlbvh.lbvh_e_separate_cases
    self.__lbvh_e_separate_cases.argtypes = [
      ctypes.c_void_p, # the object
      ctypes.c_void_p, # pp
      ctypes.c_void_p, # pe
      ctypes.c_void_p, # ee
      ctypes.c_void_p, # separated counts
    ]

  @property
  def collision_pairs(self) -> gpuarray.GPUArray:
    return self.__collision_pairs

  @property
  def collision_pairs_ccd(self) -> gpuarray.GPUArray:
    return self.__collision_pairs_ccd

  @property
  def cp_num(self) -> gpuarray.GPUArray:
    return self.__cp_num


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

  def construct_full_ccd_edges(self, vertices: gpuarray.GPUArray, moving_directions: gpuarray.GPUArray, alpha: float):
    c_alpha = ctypes.c_double(alpha)
    alpha_p = ctypes.byref(c_alpha)
    self.__lbvh_e_construct_full_ccd(
      self.__bvh_e,
      self.__to_void_p(vertices),
      self.__to_void_p(moving_directions),
      alpha_p
    )

  def cd_edges(self, vertices: gpuarray.GPUArray, dhat: float):
    self.construct_edges(vertices)
    self.__collision_pairs.fill(0)
    self.__cp_num.fill(0)
    self.__lbvh_e_self_collision_detect(
      self.__bvh_e,
      ctypes.c_double(dhat)
    )

  def ccd_edges(self, vertices: gpuarray.GPUArray, dhat: float, moving_directions: gpuarray.GPUArray, alpha: float):
    c_alpha = ctypes.c_double(alpha)
    alpha_p = ctypes.byref(c_alpha)
    self.construct_full_ccd_edges(vertices, moving_directions, alpha)
    self.__collision_pairs_ccd.fill(0)
    self.__cp_num.fill(0)
    self.__lbvh_e_self_collision_full_detect(
      self.__bvh_e,
      ctypes.c_double(dhat),
      self.__to_void_p(moving_directions),
      alpha_p
    )

  def init_faces(self,
    vertices: gpuarray.GPUArray, # list of double
    faces: gpuarray.GPUArray, # list of int3
    surface_vertices: gpuarray.GPUArray, # list of uint32
    face_num: int, # number of faces
  ):
    self.__lbvh_f_init(
      self.__bvh_f,
      self.__to_void_p(self.__btypes),
      self.__to_void_p(vertices),
      self.__to_void_p(faces),
      self.__to_void_p(surface_vertices),
      self.__to_void_p(self.__collision_pairs),
      self.__to_void_p(self.__collision_pairs_ccd),
      self.__to_void_p(self.__cp_num),
      self.__to_void_p(self.__mat_index),
      ctypes.c_int(face_num),
      ctypes.c_int(self.__num_vertices)
    )


  def construct_faces(self, vertices: gpuarray.GPUArray):
    self.__lbvh_f_construct(
      self.__bvh_f,
      self.__to_void_p(vertices)
    )

  def construct_full_ccd_faces(self, vertices: gpuarray.GPUArray, moving_directions: gpuarray.GPUArray, alpha: float):
    c_alpha = ctypes.c_double(alpha)
    alpha_p = ctypes.byref(c_alpha)
    self.__lbvh_f_construct_full_ccd(
      self.__bvh_f,
      self.__to_void_p(vertices),
      self.__to_void_p(moving_directions),
      alpha_p
    )

  def cd_faces(self, vertices: gpuarray.GPUArray, dhat: float):
    self.construct_faces(vertices)
    # empty the collision pairs
    self.__collision_pairs.fill(0)
    self.__cp_num.fill(0)
    self.__lbvh_f_self_collision_detect(
      self.__bvh_f,
      ctypes.c_double(dhat)
    )
    self.__pp.fill(0)
    self.__pe.fill(0)
    self.__pt.fill(0)
    self.__separated_counts.fill(0)
    self.__lbvh_f_separate_cases(
      self.__bvh_f,
      self.__to_void_p(self.__pp),
      self.__to_void_p(self.__pe),
      self.__to_void_p(self.__pt),
      self.__to_void_p(self.__separated_counts)
    )

  def ccd_faces(self, vertices: gpuarray.GPUArray, dhat: float, moving_directions: gpuarray.GPUArray, alpha: float):
    c_alpha = ctypes.c_double(alpha)
    alpha_p = ctypes.byref(c_alpha)
    self.__collision_pairs_ccd.fill(0)
    self.__cp_num.fill(0)
    self.construct_full_ccd_faces(vertices, moving_directions, alpha)
    self.__lbvh_f_self_collision_full_detect(
      self.__bvh_f,
      ctypes.c_double(dhat),
      self.__to_void_p(moving_directions),
      alpha_p
    )
    self.__pp.fill(0)
    self.__pe.fill(0)
    self.__pt.fill(0)
    self.__separated_counts.fill(0)
    self.__lbvh_f_separate_cases(
      self.__bvh_f,
      self.__to_void_p(self.__pp),
      self.__to_void_p(self.__pe),
      self.__to_void_p(self.__pt),
      self.__to_void_p(self.__separated_counts)
    )
    print("checking cases")
    print(self.__pt.get())
    print(self.__pe.get())
    print(self.__pp.get())



x = CCD(4)
points = gpuarray.to_gpu(np.array([[1.0, 0.0, 0.0, -0.5, 0.0, 0.866, -0.5, 0.0, -0.866, 0.5, 1.0, 0.0]]).astype(np.float64))
moving_directions = gpuarray.to_gpu(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, -1, 0.0]]).astype(np.float64))
faces = gpuarray.to_gpu(np.array([0, 1, 2, 1, 2, 3]).astype(np.int32))
surface_vertices = gpuarray.to_gpu(np.array([0, 1, 2, 3]).astype(np.uint32))
face_num = 2
x.init_faces(points, faces, surface_vertices, face_num)
x.cd_faces(points, 3)
print(x.cp_num.get())
print(x.collision_pairs.get())
print(x.collision_pairs_ccd.get())
x.ccd_faces(points, 0.01, moving_directions, 1.0)
print(x.cp_num.get())
print(x.collision_pairs.get())
print(x.collision_pairs_ccd.get())
