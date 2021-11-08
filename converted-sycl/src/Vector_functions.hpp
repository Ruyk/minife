#ifndef _Vector_functions_hpp_
#define _Vector_functions_hpp_

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
#include <vector>
#include <sstream>
#include <fstream>
#include <iomanip>
#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <TypeTraits.hpp>
#include <Vector.hpp>
#include <CudaUtils.h>
#include <cmath>

#include <algorithm>

#include <chrono>

namespace miniFE {

template<typename VectorType>
void write_vector(const std::string& filename,
                  const VectorType& vec)
{
  vec.copyToHost();
  int numprocs = 1, myproc = 0;
#ifdef HAVE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
#endif

  std::ostringstream osstr;
  osstr << filename << "." << numprocs << "." << myproc;
  std::string full_name = osstr.str();
  std::ofstream ofs(full_name.c_str());

  typedef typename VectorType::ScalarType ScalarType;

  const std::vector<ScalarType>& coefs = vec.coefs;
  for(int p=0; p<numprocs; ++p) {
    if (p == myproc) {
      if (p == 0) {
        ofs << vec.local_size << std::endl;
      }
  
      typename VectorType::GlobalOrdinalType first = vec.startIndex;
      for(size_t i=0; i<vec.local_size; ++i) {
        ofs << first+i << " " << coefs[i] << std::endl;
      }
    }
#ifdef HAVE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
  }
}

template<typename VectorType>
__inline__
void sum_into_vector_cuda(size_t num_indices,
                     const typename VectorType::GlobalOrdinalType* __restrict__ indices,
                     const typename VectorType::ScalarType* __restrict__ coefs,
                     VectorType& vec)
{

 typename VectorType::GlobalOrdinalType first = vec.startIndex;
#pragma unroll
  for(size_t i=0; i<num_indices; ++i) {
    size_t idx = indices[i] - first;

    if (idx >= vec.n) continue;

    miniFEAtomicAdd(&vec.coefs[idx], coefs[i]);
    //vec[idx] += coefs[i];
  }
}


template<typename VectorType>
void sum_into_vector(size_t num_indices,
                     const typename VectorType::GlobalOrdinalType* indices,
                     const typename VectorType::ScalarType* coefs,
                     VectorType& vec)
{
  typedef typename VectorType::GlobalOrdinalType GlobalOrdinal;
  typedef typename VectorType::ScalarType Scalar;

  GlobalOrdinal first = vec.startIndex;
  GlobalOrdinal last = first + vec.local_size - 1;

  std::vector<Scalar>& vec_coefs = vec.coefs;

  for(size_t i=0; i<num_indices; ++i) {
    if (indices[i] < first || indices[i] > last) continue;
    size_t idx = indices[i] - first;
    vec_coefs[idx] += coefs[i];
  }
}

//------------------------------------------------------------
//Compute the update of a vector with the sum of two scaled vectors where:
//
// w = alpha*x + beta*y
//
// x,y - input vectors
//
// alpha,beta - scalars applied to x and y respectively
//
// w - output vector
//
template <typename VectorType> 
void waxpby_kernel(typename VectorType::ScalarType alpha, const VectorType x, 
                               typename VectorType::ScalarType beta, const VectorType y, 
                               VectorType w, sycl::nd_item<3> item_ct1) {

  for (int idx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                 item_ct1.get_local_id(2);
       idx < x.n;
       idx += item_ct1.get_group_range(2) * item_ct1.get_local_range().get(2))
  {
      w.coefs[idx] = alpha * x.coefs[idx] + beta * y.coefs[idx];
  }
}

template<typename VectorType>
void
  waxpby(typename VectorType::ScalarType alpha, const VectorType& x,
         typename VectorType::ScalarType beta, const VectorType& y,
         VectorType& w)
{
  typedef typename VectorType::ScalarType ScalarType;
  
#ifdef MINIFE_DEBUG
  if (y.local_size < x.local_size || w.local_size < x.local_size) {
    std::cerr << "miniFE::waxpby ERROR, y and w must be at least as long as x." << std::endl;
    return;
  }
#endif
  int n = x.coefs.size();
  int BLOCK_SIZE=256;
  int BLOCKS = std::min((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 2048 * 16);

  /*
  DPCT1049:11: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  CudaManager::s1->submit([&](sycl::handler &cgh) {
    auto x_getPOD_ct1 = x.getPOD();
    auto y_getPOD_ct3 = y.getPOD();
    auto w_getPOD_ct4 = w.getPOD();

    cgh.parallel_for<dpct_kernel_name<class waxpby_kernel_cd0e25,
                                      VectorType /*Fix the type mannually*/>>(
        sycl::nd_range<3>(sycl::range<3>(1, 1, BLOCKS) *
                              sycl::range<3>(1, 1, BLOCK_SIZE),
                          sycl::range<3>(1, 1, BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
          waxpby_kernel(alpha, x_getPOD_ct1, beta, y_getPOD_ct3, w_getPOD_ct4,
                        item_ct1);
        });
  });
  cudaCheckError();
}

//-----------------------------------------------------------
//Compute the dot product of two vectors where:
//
// x,y - input vectors
//
// result - return-value
//
template<typename Vector>
void dot_kernel(const Vector x, const Vector y, typename TypeTraits<typename Vector::ScalarType>::magnitude_type *d,
                sycl::nd_item<3> item_ct1, volatile typename TypeTraits<typename Vector::ScalarType>::magnitude_type  *red) {

  typedef typename TypeTraits<typename Vector::ScalarType>::magnitude_type magnitude;
  const int BLOCK_SIZE=512; 

  magnitude sum=0;
  for (int idx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                 item_ct1.get_local_id(2);
       idx < x.n;
       idx += item_ct1.get_group_range(2) * item_ct1.get_local_range().get(2)) {
    sum+=x.coefs[idx]*y.coefs[idx];
  }

  //Do a shared memory reduction on the dot product

  red[item_ct1.get_local_id(2)] = sum;
  //__syncthreads(); if(threadIdx.x<512) {sum+=red[threadIdx.x+512]; red[threadIdx.x]=sum;}
  /*
  DPCT1065:12: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier(); if (item_ct1.get_local_id(2) < 256) {
      sum += red[item_ct1.get_local_id(2) + 256];
      red[item_ct1.get_local_id(2)] = sum;
  }
  /*
  DPCT1065:13: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier(); if (item_ct1.get_local_id(2) < 128) {
      sum += red[item_ct1.get_local_id(2) + 128];
      red[item_ct1.get_local_id(2)] = sum;
  }
  /*
  DPCT1065:14: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier(); if (item_ct1.get_local_id(2) < 64) {
      sum += red[item_ct1.get_local_id(2) + 64];
      red[item_ct1.get_local_id(2)] = sum;
  }
  /*
  DPCT1065:15: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier(); if (item_ct1.get_local_id(2) < 32) {
      sum += red[item_ct1.get_local_id(2) + 32];
      red[item_ct1.get_local_id(2)] = sum;
  }
  //the remaining ones don't need syncthreads because they are warp synchronous
                   if (item_ct1.get_local_id(2) < 16) {
                       sum += red[item_ct1.get_local_id(2) + 16];
                       red[item_ct1.get_local_id(2)] = sum;
                   }
                   if (item_ct1.get_local_id(2) < 8) {
                       sum += red[item_ct1.get_local_id(2) + 8];
                       red[item_ct1.get_local_id(2)] = sum;
                   }
                   if (item_ct1.get_local_id(2) < 4) {
                       sum += red[item_ct1.get_local_id(2) + 4];
                       red[item_ct1.get_local_id(2)] = sum;
                   }
                   if (item_ct1.get_local_id(2) < 2) {
                       sum += red[item_ct1.get_local_id(2) + 2];
                       red[item_ct1.get_local_id(2)] = sum;
                   }
                   if (item_ct1.get_local_id(2) < 1) {
                       sum += red[item_ct1.get_local_id(2) + 1];
                   }

  //save partial dot products
  if (item_ct1.get_local_id(2) == 0) d[item_ct1.get_group(2)] = sum;
}

template<typename Scalar>
void dot_final_reduce_kernel(Scalar *d, sycl::nd_item<3> item_ct1,
                             volatile Scalar *red) {
  const int BLOCK_SIZE=1024;
  Scalar sum = d[item_ct1.get_local_id(2)];

  red[item_ct1.get_local_id(2)] = sum;
  /*
  DPCT1065:16: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier(); if (item_ct1.get_local_id(2) < 512) {
      sum += red[item_ct1.get_local_id(2) + 512];
      red[item_ct1.get_local_id(2)] = sum;
  }
  /*
  DPCT1065:17: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier(); if (item_ct1.get_local_id(2) < 256) {
      sum += red[item_ct1.get_local_id(2) + 256];
      red[item_ct1.get_local_id(2)] = sum;
  }
  /*
  DPCT1065:18: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier(); if (item_ct1.get_local_id(2) < 128) {
      sum += red[item_ct1.get_local_id(2) + 128];
      red[item_ct1.get_local_id(2)] = sum;
  }
  /*
  DPCT1065:19: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier(); if (item_ct1.get_local_id(2) < 64) {
      sum += red[item_ct1.get_local_id(2) + 64];
      red[item_ct1.get_local_id(2)] = sum;
  }
  /*
  DPCT1065:20: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier(); if (item_ct1.get_local_id(2) < 32) {
      sum += red[item_ct1.get_local_id(2) + 32];
      red[item_ct1.get_local_id(2)] = sum;
  }
  //the remaining ones don't need syncthreads because they are warp synchronous
                   if (item_ct1.get_local_id(2) < 16) {
                       sum += red[item_ct1.get_local_id(2) + 16];
                       red[item_ct1.get_local_id(2)] = sum;
                   }
                   if (item_ct1.get_local_id(2) < 8) {
                       sum += red[item_ct1.get_local_id(2) + 8];
                       red[item_ct1.get_local_id(2)] = sum;
                   }
                   if (item_ct1.get_local_id(2) < 4) {
                       sum += red[item_ct1.get_local_id(2) + 4];
                       red[item_ct1.get_local_id(2)] = sum;
                   }
                   if (item_ct1.get_local_id(2) < 2) {
                       sum += red[item_ct1.get_local_id(2) + 2];
                       red[item_ct1.get_local_id(2)] = sum;
                   }
                   if (item_ct1.get_local_id(2) < 1) {
                       sum += red[item_ct1.get_local_id(2) + 1];
                   }

  //save final dot product at the front
                   if (item_ct1.get_local_id(2) == 0) d[0] = sum;
}

template<typename Vector>
typename TypeTraits<typename Vector::ScalarType>::magnitude_type
  dot(const Vector& x,
      const Vector& y)
{
  typedef typename Vector::ScalarType Scalar;
  typedef typename TypeTraits<typename Vector::ScalarType>::magnitude_type magnitude;

  int n = x.coefs.size();

#ifdef MINIFE_DEBUG
  if (y.local_size < n) {
    std::cerr << "miniFE::dot ERROR, y must be at least as long as x."<<std::endl;
    n = y.local_size;
  }
#endif

 int BLOCK_SIZE=512;
 int NUM_BLOCKS = std::min(1024, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
 static dpct::device_vector<magnitude> d(1024);
 cudaMemset_custom(dpct::get_raw_pointer(&d[0]), (magnitude)0, 1024,
                   CudaManager::s1);

 /*
 DPCT1049:22: The workgroup size passed to the SYCL kernel may exceed the limit.
 To get the device limit, query info::device::max_work_group_size. Adjust the
 workgroup size if needed.
 */
  CudaManager::s1->submit([&](sycl::handler &cgh) {
    sycl::accessor<magnitude, 1, sycl::access_mode::read_write,
                   sycl::access::target::local>
        red_acc_ct1(sycl::range<1>(512 /*BLOCK_SIZE*/), cgh);

    auto x_getPOD_ct0 = x.getPOD();
    auto y_getPOD_ct1 = y.getPOD();
    auto thrust_raw_pointer_cast_d_ct2 = dpct::get_raw_pointer(&d[0]);

    cgh.parallel_for<dpct_kernel_name<class dot_kernel_ea2530,
                                      Vector /*Fix the type mannually*/>>(
        sycl::nd_range<3>(sycl::range<3>(1, 1, NUM_BLOCKS) *
                              sycl::range<3>(1, 1, BLOCK_SIZE),
                          sycl::range<3>(1, 1, BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
          dot_kernel(x_getPOD_ct0, y_getPOD_ct1, thrust_raw_pointer_cast_d_ct2,
                     item_ct1, red_acc_ct1.get_pointer());
        });
  });
 cudaCheckError();
 /*
 DPCT1049:23: The workgroup size passed to the SYCL kernel may exceed the limit.
 To get the device limit, query info::device::max_work_group_size. Adjust the
 workgroup size if needed.
 */
  CudaManager::s1->submit([&](sycl::handler &cgh) {
    sycl::accessor<Scalar /*Fix the type mannually*/, 1,
                   sycl::access_mode::read_write, sycl::access::target::local>
        red_acc_ct1(sycl::range<1>(1024 /*BLOCK_SIZE*/), cgh);

    auto thrust_raw_pointer_cast_d_ct0 = dpct::get_raw_pointer(&d[0]);

    cgh.parallel_for<dpct_kernel_name<class dot_final_reduce_kernel_acbbb1,
                                      Scalar /*Fix the type mannually*/>>(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 1024),
                          sycl::range<3>(1, 1, 1024)),
        [=](sycl::nd_item<3> item_ct1) {
          dot_final_reduce_kernel(thrust_raw_pointer_cast_d_ct0, item_ct1,
                                  (Scalar /*Fix the type mannually*/ *)
                                      red_acc_ct1.get_pointer());
        });
  });
 cudaCheckError();
 
 static magnitude result;

 //TODO move outside?
 static bool first=true;
 if(first==true) {
   /*
   DPCT1026:24: The call to cudaHostRegister was removed because DPC++ currently
   does not support registering of existing host memory for use by device. Use
   USM to allocate memory for use by host and device.
   */
   first = false;
 }

 //TODO do this with GPU direct?
 dpct::async_dpct_memcpy(&result, dpct::get_raw_pointer(&d[0]),
                         sizeof(magnitude), dpct::device_to_host,
                         *CudaManager::s1);
 /*
 DPCT1012:21: Detected kernel execution time measurement pattern and generated
 an initial code for time measurements in SYCL. You can change the way time is
 measured depending on your goals.
 */
 auto e1_ct1 = std::chrono::steady_clock::now();
 CudaManager::e1 = CudaManager::s1->submit_barrier();
 CudaManager::e1.wait_and_throw();

#ifdef HAVE_MPI
  nvtxRangeId_t r1=nvtxRangeStartA("MPI All Reduce");
  magnitude local_dot = result, global_dot = 0;
  MPI_Datatype mpi_dtype = TypeTraits<magnitude>::mpi_type();  
  MPI_Allreduce(&local_dot, &global_dot, 1, mpi_dtype, MPI_SUM, MPI_COMM_WORLD);
  nvtxRangeEnd(r1);
  return global_dot;
#else
  return result;
#endif
}

}//namespace miniFE

#endif

