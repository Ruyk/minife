#ifndef _perform_element_loop_hpp_
#define _perform_element_loop_hpp_
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
#include <BoxIterator.hpp>
#include <simple_mesh_description.hpp>
#include <SparseMatrix_functions.hpp>
#include <box_utils.hpp>
#include <Hex8_box_utils.hpp>
#include <Hex8_ElemData.hpp>
#include <CudaHex8.hpp>
#include <cmath>

#include <algorithm>

namespace miniFE {

template<typename Scalar>
void gradients_and_psi(const Scalar* x, Scalar* gradients, Scalar *psi)
{
  //assumptions gradients has length 24 (numNodesPerElem*spatialDim)
  //        spatialDim == 3

  const Scalar u = 1.0 - x[0];
  const Scalar v = 1.0 - x[1];
  const Scalar w = 1.0 - x[2];

  const Scalar up1 = 1.0 + x[0];
  const Scalar vp1 = 1.0 + x[1];
  const Scalar wp1 = 1.0 + x[2];

//fn 0
  gradients[0] = -0.125 *  v *  w;
  gradients[1] = -0.125 *  u *  w;
  gradients[2] = -0.125 *  u *  v;
//fn 1
  gradients[3] =  0.125 *  v   *  w;
  gradients[4] = -0.125 *  up1 *  w;
  gradients[5] = -0.125 *  up1 *  v;
//fn 2
  gradients[6] =  0.125 *  vp1 *  w;
  gradients[7] =  0.125 *  up1 *  w;
  gradients[8] = -0.125 *  up1 *  vp1;
//fn 3
  gradients[9]  = -0.125 *  vp1 *  w;
  gradients[10] =  0.125 *  u   *  w;
  gradients[11] = -0.125 *  u   *  vp1;
//fn 4
  gradients[12] = -0.125 *  v   * wp1;
  gradients[13] = -0.125 *  u   * wp1;
  gradients[14] =  0.125 *  u   * v;
//fn 5
  gradients[15] =  0.125 *  v * wp1;
  gradients[16] = -0.125 *  up1 * wp1;
  gradients[17] =  0.125 *  up1 * v;
//fn 6
  gradients[18] =  0.125 *  vp1 * wp1;
  gradients[19] =  0.125 *  up1 * wp1;
  gradients[20] =  0.125 *  up1 * vp1;
//fn 7
  gradients[21] = -0.125 *  vp1 * wp1;
  gradients[22] =  0.125 *  u   * wp1;
  gradients[23] =  0.125 *  u   * vp1;

  psi[0] = 0.125 *   u *   v *   w;//(1-x)*(1-y)*(1-z)
  psi[1] = 0.125 * up1 *   v *   w;//(1+x)*(1-y)*(1-z)
  psi[2] = 0.125 * up1 * vp1 *   w;//(1+x)*(1+y)*(1-z)
  psi[3] = 0.125 *   u * vp1 *   w;//(1-x)*(1+y)*(1-z)
  psi[4] = 0.125 *   u *   v * wp1;//(1-x)*(1-y)*(1+z)
  psi[5] = 0.125 * up1 *   v * wp1;//(1+x)*(1-y)*(1+z)
  psi[6] = 0.125 * up1 * vp1 * wp1;//(1+x)*(1+y)*(1+z)
  psi[7] = 0.125 *   u * vp1 * wp1;//(1-x)*(1+y)*(1+z)
}


template<typename Scalar>
void
compute_gradients_psi(std::vector<Scalar>& gradients, std::vector<Scalar>& psi)
{
  Scalar gpts[Hex8::numGaussPointsPerDim];
  Scalar gwts[Hex8::numGaussPointsPerDim];
  gauss_pts(Hex8::numGaussPointsPerDim, gpts, gwts);
  Scalar pt[Hex8::spatialDim];
  int c=0;
  for(size_t ig=0; ig<Hex8::numGaussPointsPerDim; ++ig) {
    pt[0] = gpts[ig];
    for(size_t jg=0; jg<Hex8::numGaussPointsPerDim; ++jg) {
      pt[1] = gpts[jg];
      for(size_t kg=0; kg<Hex8::numGaussPointsPerDim; ++kg) {
        pt[2] = gpts[kg];

        assert((c+1)*Hex8::numNodesPerElem*Hex8::spatialDim<=gradients.size());
        assert((c+1)*Hex8::numNodesPerElem<=psi.size());
        gradients_and_psi(pt,&gradients[c*Hex8::numNodesPerElem*Hex8::spatialDim], &psi[c*Hex8::numNodesPerElem]);
        ++c;
      }
    }
  }
}

template<typename Scalar>
inline
void gauss_pts(int N,  Scalar* wts)
{
  //const Scalar x2 = 0.577350269; // 1.0/sqrt(3.0)
  //const Scalar x3 = 0.77459667; // sqrt(3.0/5.0)
  const Scalar w1 = 0.55555556; // 5.0/9.0
  const Scalar w2 = 0.88888889; // 8.0/9.0

  switch(N) {
  case 1:
    //pts[0] = 0.0; 
    wts[0] = 2.0;
    break;
  case 2:
    //pts[0] = -x2; 
    wts[0] = 1.0;
    // pts[1] = x2;  
    wts[1] = 1.0;
    break;
  case 3:
    //pts[0] =  -x3;  
    wts[0] = w1;
    //pts[1] =  0.0;  
    wts[1] = w2;
    //pts[2] =   x3;  
    wts[2] = w1;
    break;
  default:
    break;
  }
}

template<typename GlobalOrdinal,typename Scalar>
__inline__
void compute_element_matrix_and_vector(Scalar elem_node_coord[Hex8::spatialDim], Scalar elem_diffusion_matrix[Hex8::numNodesPerElem*Hex8::numNodesPerElem], Scalar elem_source_vector[Hex8::numNodesPerElem], const Scalar *gradients, const Scalar* psi,
                                       MINIFE_SCALAR *gauss_pts_c)
{
  Hex8::diffusionMatrix_symm(elem_node_coord, elem_diffusion_matrix, gradients,
                             gauss_pts_c);
  Hex8::sourceVector(elem_node_coord, elem_source_vector, gradients, psi,
                     gauss_pts_c);
}




template<typename MatrixType, typename VectorType, typename MeshDescription>


void element_loop_kernel(
    const MeshDescription mesh, 
    const typename MatrixType::GlobalOrdinalType num_elems, 
    const typename MatrixType::GlobalOrdinalType *elemIDs, 
    MatrixType A, 
    VectorType b,
    const typename MatrixType::ScalarType *gradients,
    const typename MatrixType::ScalarType *psi
    ,
    sycl::nd_item<3> item_ct1,
    MINIFE_SCALAR *gauss_pts_c) {
  typedef typename MatrixType::GlobalOrdinalType GlobalOrdinal;
  typedef typename MatrixType::ScalarType Scalar;

  for (int eleidx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                    item_ct1.get_local_id(2);
       eleidx < num_elems; eleidx += item_ct1.get_local_range().get(2) *
                                     item_ct1.get_group_range(2)) {

    GlobalOrdinal l_node_ids[Hex8::numNodesPerElem];
    Scalar l_diffusion_matrix[Hex8::numNodesPerElem*(Hex8::numNodesPerElem+1)/2];
    Scalar l_source_vector[Hex8::numNodesPerElem];
    Scalar l_node_coord[Hex8::numNodesPerElem*Hex8::spatialDim];

    GlobalOrdinal *node_ids=l_node_ids;
    Scalar *diffusion_matrix=l_diffusion_matrix;
    Scalar *source_vector=l_source_vector;
    Scalar *node_coords=l_node_coord;

    //Given an element-id, populate elem_data with the
    //element's node_ids and nodal-coords:
  
    get_elem_coords(mesh, elemIDs[eleidx],node_coords);

    compute_element_matrix_and_vector<GlobalOrdinal, Scalar>(
        node_coords, diffusion_matrix, source_vector, gradients, psi,
        gauss_pts_c);

    get_elem_node_ids(mesh, elemIDs[eleidx],node_ids);

    sum_into_global_linear_system_cuda(node_ids,  diffusion_matrix, source_vector, A, b);

#if 0    
    if(eleidx==123) {
      printf("\nNodeCoords: ");
      for(int i=0;i<Hex8::numNodesPerElem*Hex8::spatialDim;i++)
        printf("%lg ",node_coords[i]);
      printf("\n");
      printf("\nSourceVector: ");
      for(int i=0;i<Hex8::numNodesPerElem;i++)
        printf("%lg ",source_vector[i]);
      printf("\n");
      printf("\nNodeIDs: ");
      for(int i=0;i<Hex8::numNodesPerElem;i++)
        printf("%d ",node_ids[i]);
      printf("\n");
      printf("\ndiffusionMatrix: ");
      for(int i=0;i<Hex8::numNodesPerElem*(Hex8::numNodesPerElem+1)/2;i++)
        printf("%lg ",diffusion_matrix[i]);
      printf("\n");
    }
#endif

  }

}

template<typename MatrixType, typename VectorType>
void 
perform_element_loop_cuda(const simple_mesh_description<typename MatrixType::GlobalOrdinalType>& mesh,
                     const Box& local_elem_box,
                     MatrixType& A, VectorType& b,
                     Parameters& /*params*/
                     )
{

  typedef typename MatrixType::ScalarType Scalar;
  typedef typename MatrixType::GlobalOrdinalType GlobalOrdinal;
  typedef typename miniFE::simple_mesh_description<GlobalOrdinal> MeshType;

  int global_elems_x = mesh.global_box[0][1];
  int global_elems_y = mesh.global_box[1][1];
  int global_elems_z = mesh.global_box[2][1];

  //We will iterate the local-element-box (local portion of the mesh), and
  //get element-IDs in preparation for later assembling the FE operators
  //into the global sparse linear-system.

  GlobalOrdinal num_elems = get_num_ids<GlobalOrdinal>(local_elem_box);
  std::vector<GlobalOrdinal> elemIDs(num_elems);

  BoxIterator iter = BoxIterator::begin(local_elem_box);
  BoxIterator end  = BoxIterator::end(local_elem_box);

  for(size_t i=0; iter != end; ++iter, ++i) {
    elemIDs[i] = get_id<GlobalOrdinal>(global_elems_x, global_elems_y, global_elems_z,
        iter.x, iter.y, iter.z);
  }

  //copy elemIds to device
  GlobalOrdinal *d_elemIds;
  d_elemIds =
      (GlobalOrdinal *)dpct::dpct_malloc(sizeof(GlobalOrdinal) * num_elems);
  dpct::async_dpct_memcpy(d_elemIds, &elemIDs[0],
                          sizeof(GlobalOrdinal) * num_elems,
                          dpct::host_to_device, *CudaManager::s1);

  std::vector<Scalar> gradients(Hex8::numGaussPointsPerDim * Hex8::numGaussPointsPerDim * Hex8::numGaussPointsPerDim * Hex8::numNodesPerElem * Hex8::spatialDim);
  std::vector<Scalar> psi(Hex8::numNodesPerElem*Hex8::numNodesPerElem);
  Scalar gp[Hex8::numGaussPointsPerDim];

  dpct::device_vector<Scalar> d_gradients(
      Hex8::spatialDim * Hex8::numNodesPerElem * Hex8::numGaussPointsPerDim *
      Hex8::numGaussPointsPerDim * Hex8::numGaussPointsPerDim);
  dpct::device_vector<Scalar> d_psi(
      Hex8::numNodesPerElem * Hex8::numGaussPointsPerDim *
      Hex8::numGaussPointsPerDim * Hex8::numGaussPointsPerDim);

  //precompute gradients, psi, and gauss pts
  compute_gradients_psi<Scalar>(gradients, psi);
  gauss_pts(Hex8::numGaussPointsPerDim,gp);

  //copy gradients and psi to device memory
  dpct::async_dpct_memcpy(dpct::get_raw_pointer(&d_gradients[0]), &gradients[0],
                          sizeof(Scalar) * gradients.size(),
                          dpct::host_to_device, *CudaManager::s1);
  dpct::async_dpct_memcpy(dpct::get_raw_pointer(&d_psi[0]), &psi[0],
                          sizeof(Scalar) * psi.size(), dpct::host_to_device,
                          *CudaManager::s1);

  //copy gauss_pts to constant memory
  dpct::async_dpct_memcpy(Hex8::gauss_pts_c.get_ptr(*CudaManager::s1), gp, sizeof(gp),
                          dpct::host_to_device, *CudaManager::s1);
  cudaCheckError();

  //initialize the source vector to 0's
  std::fill(oneapi::dpl::execution::make_device_policy<class Policy_2214bd>(
                dpct::get_default_queue()),
            b.d_coefs.begin(), b.d_coefs.end(), 0);

  const int BLOCK_SIZE=128;
  const int NUM_BLOCKS = std::min((num_elems + BLOCK_SIZE - 1) / BLOCK_SIZE,
                                  896); // 64 blocks per SM

  /*
  DPCT1007:48: Migration of this CUDA API is not supported by the Intel(R) DPC++
  Compatibility Tool.
  */
  // cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
  /*
  DPCT1026:49: The call to cudaDeviceSetSharedMemConfig was removed because
  DPC++ currently does not support configuring shared memory on devices.
  */

  //call finite-element assembly kernel:
  CudaManager::s1->submit([&](sycl::handler &cgh) {
    Hex8::gauss_pts_c.init(*CudaManager::s1);

    auto gauss_pts_c_acc_ct1 = Hex8::gauss_pts_c.get_access(cgh);
    dpct::access_wrapper<GlobalOrdinal *> d_elemIds_acc_ct2(d_elemIds, cgh);

    auto mesh_getPOD_ct0 = mesh.getPOD();
    auto A_getPOD_ct3 = A.getPOD();
    auto b_getPOD_ct4 = b.getPOD();
    auto thrust_raw_pointer_cast_d_gradients_ct5 =
        dpct::get_raw_pointer(&d_gradients[0]);
    auto thrust_raw_pointer_cast_d_psi_ct6 = dpct::get_raw_pointer(&d_psi[0]);

    cgh.parallel_for<dpct_kernel_name<class element_loop_kernel_b38dfa,
                                      Scalar /*Fix the type mannually*/,
                                      GlobalOrdinal /*Fix the type mannually*/,
                                      MeshType /*Fix the type mannually*/>>(
        sycl::nd_range<3>(sycl::range<3>(1, 1, NUM_BLOCKS) *
                              sycl::range<3>(1, 1, BLOCK_SIZE),
                          sycl::range<3>(1, 1, BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
          element_loop_kernel(mesh_getPOD_ct0, num_elems,
                              d_elemIds_acc_ct2.get_raw_pointer(), A_getPOD_ct3,
                              b_getPOD_ct4,
                              thrust_raw_pointer_cast_d_gradients_ct5,
                              thrust_raw_pointer_cast_d_psi_ct6, item_ct1,
                              gauss_pts_c_acc_ct1.get_pointer());
        });
  });

  cudaCheckError();

}
}//namespace miniFE

#endif

