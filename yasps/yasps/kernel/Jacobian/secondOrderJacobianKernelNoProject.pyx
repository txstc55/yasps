from __future__ import annotations

from typing import List

from yasps.attribute import attribute
from yasps.connectivity import connectivity
from yasps.primitiveUnion import primitiveUnion


class secondOrderJacobianKernelNoProject:
  """Generate the rectangular analogue of the no-projection Hessian kernel."""

  def __init__(
    self,
    att: attribute,
    row_stride: int,
    column_stride: int,
    attribute_name: str
  ):
    sorted_datas: List[attribute] = att.deviceKernel.kernelDatas
    sorted_connectivities: List[connectivity] = (
      att.deviceKernel.kernelConnectivity
    )
    sorted_unions: List[primitiveUnion] = (
      att.deviceKernel.kernelPrimitiveUnions
    )

    self.__kernel_string = f'''
#include "allHeaders.cuh"

extern "C" {{
__global__ void compute_second_order_jacobian_no_project_global(
  {"".join([f"const double* {x.code_generation_data_name}, " for x in sorted_datas])}
  {"".join([f"const unsigned int* {x.code_generation_index_name}, " for x in sorted_connectivities])}
  {"".join([f"const unsigned int* {x.code_generation_csr_name}, " for x in sorted_connectivities if x.dimension == 0])}
  {"".join([f"const unsigned int* {x.code_generation_counts_name}, " for x in sorted_unions])}
  const unsigned int* row_indices,
  const unsigned short int* row_sizes,
  const short int* row_permutations,
  const unsigned int* column_indices,
  const unsigned short int* column_sizes,
  const short int* column_permutations,
  const unsigned int* coordinates_outer,
  const unsigned int* lookups,
  double* jacobian_blocks,
  const unsigned int num_instances
) {{
  const unsigned int instance = blockIdx.x * blockDim.x + threadIdx.x;
  if (instance >= num_instances) {{
    return;
  }}

  // The generated device function writes the complete raw rectangular
  // derivative, including any fixed-width space reserved by UNION branches.
  Eigen::Matrix<double, {att.rows}, {att.cols}{", Eigen::RowMajor" if att.cols > 1 else ""}> local_jacobian;
  {attribute_name}_device_function(
    {"".join([f"{x.code_generation_data_name}, " for x in sorted_datas])}
    {"".join([f"{x.code_generation_index_name}, " for x in sorted_connectivities])}
    {"".join([f"{x.code_generation_csr_name}, " for x in sorted_connectivities if x.dimension == 0])}
    {"".join([f"{x.code_generation_counts_name}, " for x in sorted_unions])}
    instance,
    local_jacobian.data()
  );

  // Coordinates are emitted in positive-row-slot x positive-column-slot
  // order.  Walk that same order here.  Raw offsets are reconstructed from
  // every segment size, including permutation-zero UNION padding; the
  // permutation is only used to decide whether a coordinate exists.
  unsigned int valid_block_count = 0;
  const unsigned int coordinate_start = coordinates_outer[instance];
  unsigned int row_offset = 0;
  for (unsigned int i = 0; i < {row_stride}; ++i) {{
    const unsigned int row_slot = instance * {row_stride} + i;
    const unsigned int row_size = row_sizes[row_slot];
    const bool valid_row = (
      row_permutations[row_slot] > 0 && row_indices[row_slot] >= 2
    );

    if (valid_row) {{
      unsigned int column_offset = 0;
      for (unsigned int j = 0; j < {column_stride}; ++j) {{
        const unsigned int column_slot = instance * {column_stride} + j;
        const unsigned int column_size = column_sizes[column_slot];
        const bool valid_column = (
          column_permutations[column_slot] > 0 &&
          column_indices[column_slot] >= 2
        );

        if (valid_column) {{
          const unsigned int placement =
            lookups[coordinate_start + valid_block_count];
          for (unsigned int r = 0; r < row_size; ++r) {{
            for (unsigned int c = 0; c < column_size; ++c) {{
              atomic_add_grouped(
                &jacobian_blocks[placement + r * column_size + c],
                local_jacobian(row_offset + r, column_offset + c)
              );
            }}
          }}
          ++valid_block_count;
        }}

        // Padding is absent from the coordinate stream but present in the
        // raw generated matrix, exactly as in hessianKernelNoProject.
        column_offset += column_size;
      }}
    }}

    row_offset += row_size;
  }}
}}
}}
'''

  @property
  def kernelString(self) -> str:
    return self.__kernel_string
