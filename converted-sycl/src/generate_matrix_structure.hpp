#ifndef _generate_matrix_structure_hpp_
#define _generate_matrix_structure_hpp_

//@HEADER
// ************************************************************************
// 
//               HPCCG: Simple Conjugate Gradient Benchmark Code
//                 Copyright (2006) Sandia Corporation
// 
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
// 
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//  
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//  
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
// Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
// 
// ************************************************************************
//@HEADER

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <sstream>
#include <stdexcept>
#include <map>
#include <algorithm>

#include <simple_mesh_description.hpp>
#include <SparseMatrix_functions.hpp>
#include <box_utils.hpp>
#include <utils.hpp>
#include <CudaUtils.h>
#ifdef HAVE_MPI
#include <mpi.h>
#include <chrono>

#include <cmath>

#endif

namespace miniFE {

template<class T> __inline__ void swap(T &a, T &b) {
  T c=a;
  a=b;
  b=c;
}

template<typename MeshType, typename MatrixType> 
void
generate_matrix_structure_kernel(MeshType mesh, MatrixType A, int size_x, int size_y, int size_z,
                                 sycl::nd_item<3> item_ct1) {
  typedef typename MatrixType::GlobalOrdinalType GlobalOrdinal;

  int global_nodes_x = mesh.global_elems_x+1;
  int global_nodes_y = mesh.global_elems_y+1;
  int global_nodes_z = mesh.global_elems_z+1;
  GlobalOrdinal global_nrows= global_nodes_x*global_nodes_y*global_nodes_z;

  GlobalOrdinal num_elements=size_x*size_y*size_z;

  for (GlobalOrdinal eleidx =
           item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
           item_ct1.get_local_id(2);
       eleidx < num_elements; eleidx += item_ct1.get_group_range(2) *
                                        item_ct1.get_local_range().get(2)) {
    //compute element coordinates
    int ix=eleidx%size_x+mesh.local_box[0][0];
    int iy=eleidx/size_x%size_y+mesh.local_box[1][0];
    int iz=eleidx/size_x/size_y%size_z+mesh.local_box[2][0];

    GlobalOrdinal row_id=get_id<GlobalOrdinal>(global_nodes_x, global_nodes_y, global_nodes_z, ix, iy, iz);
    A.rows[eleidx]= mesh.map_ids_to_rows.find_row_for_id(row_id);

    GlobalOrdinal col = -1;

    unsigned int cols[27];

#pragma unroll
    for(int nodeIdx=0;nodeIdx<27;nodeIdx++) {

      //compute neighbor offsets
      int sx=nodeIdx%3-1;
      int sy=nodeIdx/3%3-1;
      int sz=nodeIdx/3/3%3-1;

      GlobalOrdinal col_id=get_id<GlobalOrdinal>(global_nodes_x, global_nodes_y, global_nodes_z, ix+sx, iy+sy, iz+sz);

      //if col_id is in range
      if (col_id >= 0 && col_id < global_nrows) 
        //compute column
        col = mesh.map_ids_to_rows.find_row_for_id(col_id);

      cols[nodeIdx]=col;
      A.coefs[eleidx+nodeIdx*A.pitch]=0;
    }

    //sort columns in registers
#pragma unroll
    for(int i=0;i<27;i++) {
#pragma unroll
      for(int j=0;j<27;j++) {
        if(i<j) {
          if(cols[j]<cols[i])
            swap(cols[i],cols[j]);
        }
      }
    }
     
    //write columns
    #pragma unroll
    for(int i=0;i<27;i++) {
      A.cols[eleidx+i*A.pitch]=cols[i];
      //if(i>1) assert(cols[i]>cols[i+1]);

    }
  }
}

//x dim = warp , 1 warp per element(row), 1 thread per non-zero(col)
template<typename MeshType, typename MatrixType> 
void
generate_matrix_structure_kernel_old(MeshType mesh, MatrixType A, int size_x, int size_y, int size_z,
                                     sycl::nd_item<3> item_ct1) {
  typedef typename MatrixType::GlobalOrdinalType GlobalOrdinal;

  int nodeIdx = item_ct1.get_local_id(2);

  int global_nodes_x = mesh.global_elems_x+1;
  int global_nodes_y = mesh.global_elems_y+1;
  int global_nodes_z = mesh.global_elems_z+1;
  GlobalOrdinal global_nrows= global_nodes_x*global_nodes_y*global_nodes_z;

  GlobalOrdinal num_elements=size_x*size_y*size_z;

  for (GlobalOrdinal eleidx =
           item_ct1.get_group(2) * item_ct1.get_local_range().get(1) +
           item_ct1.get_local_id(1);
       eleidx < num_elements; eleidx += item_ct1.get_group_range(2) *
                                        item_ct1.get_local_range().get(1)) {
    //compute element coordinates
    int ix=eleidx%size_x+mesh.local_box[0][0];
    int iy=eleidx/size_x%size_y+mesh.local_box[1][0];
    int iz=eleidx/size_x/size_y%size_z+mesh.local_box[2][0];

    GlobalOrdinal row_id=get_id<GlobalOrdinal>(global_nodes_x, global_nodes_y, global_nodes_z, ix, iy, iz);
    A.rows[eleidx]= mesh.map_ids_to_rows.find_row_for_id(row_id);

    GlobalOrdinal col = -1;

    if(nodeIdx<27) {
      //compute neighbor offsets
      int sx=nodeIdx%3-1;
      int sy=nodeIdx/3%3-1;
      int sz=nodeIdx/3/3%3-1;

      GlobalOrdinal col_id=get_id<GlobalOrdinal>(global_nodes_x, global_nodes_y, global_nodes_z, ix+sx, iy+sy, iz+sz);

      //if col_id is in range
      if (col_id >= 0 && col_id < global_nrows) 
        //compute column
        col = mesh.map_ids_to_rows.find_row_for_id(col_id);
    }

    if(sizeof(GlobalOrdinal)==4)
      col = __sort((unsigned int)col, item_ct1);
    else
      col = __sort((unsigned long long)col, item_ct1);

    if(nodeIdx<27) {
      A.cols[eleidx+nodeIdx*A.pitch]=col;
      A.coefs[eleidx+nodeIdx*A.pitch]=0;
    }
  }
}


template<typename MatrixType>
int
generate_matrix_structure(const simple_mesh_description<typename MatrixType::GlobalOrdinalType>& mesh,
                          MatrixType& A)
{
  int myproc = 0;
#ifdef HAVE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
#endif

  int threw_exc = 0;
  try {

    typedef typename MatrixType::GlobalOrdinalType GlobalOrdinal;
    typedef typename MatrixType::LocalOrdinalType LocalOrdinal;

    int global_nodes_x = mesh.global_box[0][1]+1;
    int global_nodes_y = mesh.global_box[1][1]+1;
    int global_nodes_z = mesh.global_box[2][1]+1;
    Box box;
    copy_box(mesh.local_box, box);

    //num-owned-nodes in each dimension is num-elems+1
    //only if num-elems > 0 in that dimension *and*
    //we are at the high end of the global range in that dimension:
    if (box[0][1] > box[0][0] && box[0][1] == mesh.global_box[0][1]) ++box[0][1];
    if (box[1][1] > box[1][0] && box[1][1] == mesh.global_box[1][1]) ++box[1][1];
    if (box[2][1] > box[2][0] && box[2][1] == mesh.global_box[2][1]) ++box[2][1];

    GlobalOrdinal global_nrows = global_nodes_x;
    global_nrows *= global_nodes_y*global_nodes_z;

    GlobalOrdinal nrows = get_num_ids<GlobalOrdinal>(box);
    try {
      nvtxRangeId_t r1=nvtxRangeStartA("reserve space in A");
      A.reserve_space(nrows, 27);
      nvtxRangeEnd(r1);
    }
    catch(std::exception& exc) {
      std::ostringstream osstr;
      osstr << "One of A.rows.resize, A.row_offsets.resize, A.packed_cols.reserve or A.packed_coefs.reserve: nrows=" <<nrows<<": ";
      osstr << exc.what();
      std::string str1 = osstr.str();
      throw std::runtime_error(str1);
    }


    int size_x=box[0][1]-box[0][0];
    int size_y=box[1][1]-box[1][0];
    int size_z=box[2][1]-box[2][0];
    int num_elems=size_x*size_y*size_z;
#if 0
    dim3 BLOCK_SIZE;
    BLOCK_SIZE.x=32;
    BLOCK_SIZE.y=8;
    int NUM_BLOCKS=min((num_elems+BLOCK_SIZE.y-1)/BLOCK_SIZE.y,448); 
    generate_matrix_structure_kernel_old<<<NUM_BLOCKS,BLOCK_SIZE,0,CudaManager::s1>>>(mesh.getPOD(), A.getPOD(), size_x, size_y, size_z);
    cudaCheckError();
#else
    int BLOCK_SIZE=128;
    int NUM_BLOCKS =
        std::min((num_elems + BLOCK_SIZE - 1) / BLOCK_SIZE, 448 * 32);
    /*
    DPCT1049:36: The workgroup size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the workgroup size if needed.
    */
    CudaManager::s1->submit([&](sycl::handler &cgh) {
      auto mesh_getPOD_ct0 = mesh.getPOD();
      auto A_getPOD_ct1 = A.getPOD();

      cgh.parallel_for<
          dpct_kernel_name<class generate_matrix_structure_kernel_681844,
                           GlobalOrdinal /*Fix the type mannually*/,
                           LocalOrdinal /*Fix the type mannually*/>>(
          sycl::nd_range<3>(sycl::range<3>(1, 1, NUM_BLOCKS) *
                                sycl::range<3>(1, 1, BLOCK_SIZE),
                            sycl::range<3>(1, 1, BLOCK_SIZE)),
          [=](sycl::nd_item<3> item_ct1) {
            generate_matrix_structure_kernel(mesh_getPOD_ct0, A_getPOD_ct1,
                                             size_x, size_y, size_z, item_ct1);
          });
    });
    cudaCheckError();

#endif
    //Allocate host arrays
    nvtxRangeId_t r2=nvtxRangeStartA("allocate host memory");
    A.rows.resize(A.d_rows.size());
    A.cols.resize(A.d_cols.size());
    /*
    DPCT1026:37: The call to cudaHostRegister was removed because DPC++
    currently does not support registering of existing host memory for use by
    device. Use USM to allocate memory for use by host and device.
    */
    cudaCheckError();
    nvtxRangeEnd(r2);

    //TODO see where rows is needed and verify it is...
    //copy rows back to host, this is needed elsewhere
    dpct::async_dpct_memcpy(&A.rows[0], dpct::get_raw_pointer(&A.d_rows[0]),
                            sizeof(GlobalOrdinal) * A.rows.size(),
                            dpct::device_to_host, *CudaManager::s1);
    dpct::async_dpct_memcpy(&A.cols[0], dpct::get_raw_pointer(&A.d_cols[0]),
                            sizeof(GlobalOrdinal) * A.cols.size(),
                            dpct::device_to_host, *CudaManager::s1);
    cudaCheckError();
    /*
    DPCT1012:35: Detected kernel execution time measurement pattern and
    generated an initial code for time measurements in SYCL. You can change the
    way time is measured depending on your goals.
    */
    e1_ct1 = std::chrono::steady_clock::now();
    CudaManager::e1 = CudaManager::s1->submit_barrier();
  }
  catch(...) {
    std::cout << "proc " << myproc << " threw an exception in generate_matrix_structure, probably due to running out of memory." << std::endl;
    threw_exc = 1;
  }

#ifdef HAVE_MPI
  int global_throw = 0;
  MPI_Allreduce(&threw_exc, &global_throw, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  threw_exc = global_throw;
#endif
  if (threw_exc) {
    return 1;
  }

  return 0;
}

}//namespace miniFE

#endif

