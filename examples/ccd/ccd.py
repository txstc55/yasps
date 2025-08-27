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
import time

class CCD:
  def __init__(self, num_vertices: int, all_vertices: int, max_cd_pairs: int = 10000000, max_ccd_pairs: int = 100000000, mesh_indices: List[int] = []):
    module_dir = os.path.dirname(os.path.abspath(__file__))  # always resolves to y.py's directory
    mlbvh_so_path = os.path.join(module_dir, "libmlbvh.so")
    accd_so_path = os.path.join(module_dir, "libaccd.so")
    if not os.path.exists(mlbvh_so_path) or not os.path.exists(accd_so_path):
      # we first compile the code
      mlbvh_cu = os.path.join(module_dir, "mlbvh.cu")
      eigen_cu = os.path.join(module_dir, "gpu_eigen_libs.cu")
      mlbvh_o = os.path.join(module_dir, "mlbvh.o")
      eigen_o = os.path.join(module_dir, "gpu_eigen_libs.o")
      accd_cu = os.path.join(module_dir, "ACCD.cu")
      accd_o = os.path.join(module_dir, "ACCD.o")

      compile_cmds = [
        [
          "nvcc", "-std=c++17", "-Xcompiler", "-fPIC",
          "-O3",
          "-I/usr/include/eigen", "-I.",
          "-gencode", "arch=compute_86,code=sm_86",
          "--relocatable-device-code=true",
          "-c", mlbvh_cu, "-o", mlbvh_o
        ],
        [
          "nvcc", "-std=c++17", "-Xcompiler", "-fPIC",
          "-O3",
          "-I/usr/include/eigen", "-I.",
          "-gencode", "arch=compute_86,code=sm_86",
          "--relocatable-device-code=true",
          "-c", accd_cu, "-o", accd_o
        ],
        [
          "nvcc", "-std=c++17", "-Xcompiler", "-fPIC",
          "-O3",
          "-I/usr/include/eigen", "-I.",
          "-gencode", "arch=compute_86,code=sm_86",
          "--relocatable-device-code=true",
          "-c", eigen_cu, "-o", eigen_o
        ],
        [
          "nvcc", "-std=c++17", "-Xcompiler", "-fPIC",
          "-O3",
          "-gencode", "arch=compute_86,code=sm_86",
          "--relocatable-device-code=true",
          mlbvh_o, eigen_o,
          "-o", mlbvh_so_path, "--shared"
        ],
        [
          "nvcc", "-std=c++17", "-Xcompiler", "-fPIC",
          "-O3",
          "-gencode", "arch=compute_86,code=sm_86",
          "--relocatable-device-code=true",
          accd_o, eigen_o,
          "-o", accd_so_path, "--shared"
        ]
      ]

      for cmd in compile_cmds:
        # print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    # we are probably certain that the library is compiled
    # first get the file
    self.__mlbvh = ctypes.CDLL(mlbvh_so_path)
    self.__accd = ctypes.CDLL(accd_so_path)

    self.__self_largestFeasibleStepSize = self.__accd.self_largestFeasibleStepSize
    self.__self_largestFeasibleStepSize.argtypes = [
      ctypes.c_double, # slackness
      ctypes.c_void_p, # vertices, gpu array pointer to double3
      ctypes.c_void_p, # collision pairs, gpu array pointer to int4
      ctypes.c_void_p, # moving directions, gpu array pointer to double3
      ctypes.c_void_p, # mqueue, gpu array pointer to double
      ctypes.c_int # number of collision pairs
    ]
    self.__self_largestFeasibleStepSize.restype = ctypes.c_double


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
      ctypes.c_void_p, # the mesh index, we use this to avoid self collision detection if needed
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
      ctypes.c_void_p, # the mesh index, we use this to avoid self collision detection if needed
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
    self.__cp_num = gpuarray.to_gpu(np.zeros(5, dtype=np.uint32)) # for some reason this is 5 in GIPC and we will keep it this way

    self.__num_vertices = num_vertices
    self.__btypes = gpuarray.to_gpu(np.zeros(all_vertices, dtype=np.int32)) # initialize empty array for btypes
    self.__pp = gpuarray.to_gpu(np.zeros((max_cd_pairs * 2), dtype=np.uint32))
    self.__pe = gpuarray.to_gpu(np.zeros((max_cd_pairs * 3), dtype=np.uint32))
    self.__pt = gpuarray.to_gpu(np.zeros((max_cd_pairs * 4), dtype=np.uint32))
    self.__ee = gpuarray.to_gpu(np.zeros((max_cd_pairs * 4), dtype=np.uint32))
    self.__separated_counts = gpuarray.to_gpu(np.zeros(4, dtype=np.uint32))
    self.__mqueue = gpuarray.to_gpu(np.zeros(max_ccd_pairs, dtype=np.float64))

    self.__mesh_indices: gpuarray.GPUArray
    if len(mesh_indices) == 0:
      self.__mesh_indices = gpuarray.to_gpu(np.zeros(num_vertices, dtype=np.uint32))
    else:
      if len(mesh_indices) != num_vertices:
        raise ValueError("Length of mesh_indices must be equal to num_vertices")
      self.__mesh_indices = gpuarray.to_gpu(np.array(mesh_indices, dtype=np.uint32))

    self.__lbvh_f_separate_cases_ccd = self.__mlbvh.lbvh_f_separate_cases_ccd
    self.__lbvh_f_separate_cases_ccd.argtypes = [
      ctypes.c_void_p, # the object
      ctypes.c_void_p, # pp
      ctypes.c_void_p, # pe
      ctypes.c_void_p, # pt
      ctypes.c_void_p, # separated counts
    ]
    self.__lbvh_e_separate_cases_ccd = self.__mlbvh.lbvh_e_separate_cases_ccd
    self.__lbvh_e_separate_cases_ccd.argtypes = [
      ctypes.c_void_p, # the object
      ctypes.c_void_p, # pp
      ctypes.c_void_p, # pe
      ctypes.c_void_p, # ee
      ctypes.c_void_p, # separated counts
    ]

    self.__lbvh_f_separate_cases_cd = self.__mlbvh.lbvh_f_separate_cases_cd
    self.__lbvh_f_separate_cases_cd.argtypes = [
      ctypes.c_void_p, # the object
      ctypes.c_void_p, # pp
      ctypes.c_void_p, # pe
      ctypes.c_void_p, # pt
      ctypes.c_void_p, # separated counts
    ]
    self.__lbvh_e_separate_cases_cd = self.__mlbvh.lbvh_e_separate_cases_cd
    self.__lbvh_e_separate_cases_cd.argtypes = [
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

  @property
  def separated_counts(self) -> list[int]:
    return self.__separated_counts.get().tolist()

  @property
  def pp(self) -> gpuarray.GPUArray:
    return self.__pp

  @property
  def pe(self) -> gpuarray.GPUArray:
    return self.__pe

  @property
  def pt(self) -> gpuarray.GPUArray:
    return self.__pt

  @property
  def ee(self) -> gpuarray.GPUArray:
    return self.__ee


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
      self.__to_void_p(self.__mesh_indices),
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
    self.__lbvh_e_self_collision_detect(
      self.__bvh_e,
      ctypes.c_double(dhat)
    )
    self.__lbvh_e_separate_cases_cd(
      self.__bvh_e,
      self.__to_void_p(self.__pp),
      self.__to_void_p(self.__pe),
      self.__to_void_p(self.__ee),
      self.__to_void_p(self.__separated_counts)
    )

  def ccd_edges(self, vertices: gpuarray.GPUArray, dhat: float, moving_directions: gpuarray.GPUArray, alpha: float):
    c_alpha = ctypes.c_double(alpha)
    alpha_p = ctypes.byref(c_alpha)
    self.construct_full_ccd_edges(vertices, moving_directions, alpha)
    self.__lbvh_e_self_collision_full_detect(
      self.__bvh_e,
      ctypes.c_double(dhat),
      self.__to_void_p(moving_directions),
      alpha_p
    )
    # self.__lbvh_e_separate_cases_ccd(
    #   self.__bvh_e,
    #   self.__to_void_p(self.__pp),
    #   self.__to_void_p(self.__pe),
    #   self.__to_void_p(self.__ee),
    #   self.__to_void_p(self.__separated_counts)
    # )

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
      self.__to_void_p(self.__mesh_indices),
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
    self.__lbvh_f_self_collision_detect(
      self.__bvh_f,
      ctypes.c_double(dhat)
    )
    self.__lbvh_f_separate_cases_cd(
      self.__bvh_f,
      self.__to_void_p(self.__pp),
      self.__to_void_p(self.__pe),
      self.__to_void_p(self.__pt),
      self.__to_void_p(self.__separated_counts)
    )

  def ccd_faces(self, vertices: gpuarray.GPUArray, dhat: float, moving_directions: gpuarray.GPUArray, alpha: float):

    c_alpha = ctypes.c_double(alpha)
    alpha_p = ctypes.byref(c_alpha)
    self.construct_full_ccd_faces(vertices, moving_directions, alpha)
    self.__lbvh_f_self_collision_full_detect(
      self.__bvh_f,
      ctypes.c_double(dhat),
      self.__to_void_p(moving_directions),
      alpha_p
    )
    # self.__lbvh_f_separate_cases_ccd(
    #   self.__bvh_f,
    #   self.__to_void_p(self.__pp),
    #   self.__to_void_p(self.__pe),
    #   self.__to_void_p(self.__pt),
    #   self.__to_void_p(self.__separated_counts)
    # )

  def reset(self):
    self.__pp.fill(0)
    self.__pe.fill(0)
    self.__pt.fill(0)
    self.__ee.fill(0)
    self.__separated_counts.fill(0)
    self.__cp_num.fill(0)
    self.__collision_pairs.fill(0)
    self.__collision_pairs_ccd.fill(0)
    self.__mqueue.fill(0)


  def cd(self, vertices: gpuarray.GPUArray, dhat: float):
    time_start = time.time()
    self.reset()
    self.cd_faces(vertices, dhat)
    self.__collision_pairs.fill(0)
    self.__cp_num.fill(0)
    self.__collision_pairs_ccd.fill(0)
    self.cd_edges(vertices, dhat)
    time_end = time.time()
    # print time in milliseconds
    print(f"Collision detection took {(time_end - time_start) * 1000:.2f} ms")

  def ccd(self, vertices: gpuarray.GPUArray, dhat: float, moving_directions: gpuarray.GPUArray, alpha: float):
    time_start = time.time()
    self.reset()
    self.ccd_faces(vertices, dhat, moving_directions, alpha)
    self.ccd_edges(vertices, dhat, moving_directions, alpha)
    time_end = time.time()
    # print time in milliseconds
    print(f"Continuous collision detection took {(time_end - time_start) * 1000:.2f} ms")

  def __del__(self):
    if self.__mlbvh is not None:
      # Free the resources
      self.__mlbvh.destroy_lbvh_f(self.__bvh_f)
      self.__mlbvh.destroy_lbvh_e(self.__bvh_e)

  def compute_largest_step_size(self, slackness, vertices: gpuarray.GPUArray, moving_directions: gpuarray.GPUArray):
    time_start = time.time()
    c_slackness = ctypes.c_double(slackness)
    step = self.__self_largestFeasibleStepSize(
      c_slackness,
      self.__to_void_p(vertices),
      self.__to_void_p(self.__collision_pairs_ccd),
      self.__to_void_p(moving_directions),
      self.__to_void_p(self.__mqueue),
      self.__cp_num.get()[0]
    )
    time_end = time.time()
    # print time in milliseconds
    print(f"Computing largest step size took {(time_end - time_start) * 1000:.2f} ms")
    return step




# x = CCD(4)
# points = gpuarray.to_gpu(np.array([[1.0, 0.0, 0.0, -0.5, 0.0, 0.866, -0.5, 0.0, -0.866, 0.5, 1.0, 0.0]]).astype(np.float64))
# moving_directions = gpuarray.to_gpu(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, -1, 0.0]]).astype(np.float64))
# faces = gpuarray.to_gpu(np.array([0, 1, 2]).astype(np.int32))
# surface_vertices = gpuarray.to_gpu(np.array([0, 1, 2, 3]).astype(np.uint32))
# face_num = 1
# x.init_faces(points, faces, surface_vertices, face_num)
# print("Face initialized")
# x.cd_faces(points, 3.0)
# print(x.cp_num.get())
# print(x.collision_pairs.get())
# print(x.collision_pairs_ccd.get())
# x.ccd_faces(points, 0.01, moving_directions, 1.0)
# print(x.cp_num.get())
# print(x.collision_pairs.get())
# print(x.collision_pairs_ccd.get())
