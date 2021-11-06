#ifndef _make_local_matrix_hpp_
#define _make_local_matrix_hpp_

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
#include <map>

#ifdef HAVE_MPI
#include <mpi.h>
#include <chrono>

#include <cmath>

#include <algorithm>

#endif

namespace miniFE {
 
template<typename GlobalOrdinal>
struct PODColMarkMap {
    GlobalOrdinal *table;
    GlobalOrdinal table_size;

    __inline__ void insert(GlobalOrdinal i) {
      GlobalOrdinal v=i;
      bool inserted=false;
      int count=0;
      while(!inserted) {
        GlobalOrdinal h=hash(v);
        inserted=try_update(h,i);
        v=h;
        assert(count++<1000);
      }
    }
    
    __inline__ bool try_update(GlobalOrdinal loc, GlobalOrdinal i) {
      /*
      DPCT1039:50: The generated code assumes that "table+loc" points to the
      global memory address space. If it points to a local memory address space,
      replace "dpct::atomic_compare_exchange_strong" with
      "dpct::atomic_compare_exchange_strong<GlobalOrdinal,
      sycl::access::address_space::local_space>".
      */
      GlobalOrdinal old =
          dpct::atomic_compare_exchange_strong(table + loc, -1, i);
      return (old==i || old==-1);
    }
    __inline__ GlobalOrdinal hash(GlobalOrdinal a) {
      a = (a + 0x7ed55d16) + (a << 12);
      a = (a ^ 0xc761c23c) + (a >> 19);
      a = (a + 0x165667b1) + (a << 5);
      a = (a ^ 0xd3a2646c) + (a << 9);
      a = (a + 0xfd7046c5) + (a << 3);
      a = (a ^ 0xb55a4f09) + (a >> 16);
      return sycl::abs(a ^ 0x4a51e590) % table_size;
    }
 };

template<typename GlobalOrdinal>
class ColMarkMap {
  public:
    ColMarkMap(GlobalOrdinal size) { 
      resize(size);
    }

    PODColMarkMap<GlobalOrdinal> getPOD() {
      PODColMarkMap<GlobalOrdinal> ret;
      ret.table = dpct::get_raw_pointer(&table[0]);
      ret.table_size=table.size();
      return ret;
    }

    void resize(GlobalOrdinal size) {
      table.resize(size);
      table.assign(size,-1);
    }
    dpct::device_vector<GlobalOrdinal> table;
};

template<typename MatrixType, typename MapType>
void markExternalColumnsInMap(MatrixType A, MapType map,
                              sycl::nd_item<3> item_ct1)
{
  //TODO also set bitmap here?
  typedef typename MatrixType::GlobalOrdinalType GlobalOrdinalType;
  for (int row_idx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                     item_ct1.get_local_id(2);
       row_idx < A.num_rows; row_idx += item_ct1.get_local_range().get(2) *
                                        item_ct1.get_group_range(2))
  {
    int offset=row_idx;
    for(int j=0;j<A.num_cols_per_row;++j) {
      GlobalOrdinalType col=A.cols[offset];
      if(col==-1) break;
      //if this column is larger than the number of rows in A it is an external
      if( col<-1 ) {
        map.insert(-col-2);
      }
      offset+=A.pitch;
    }
  }
}
template <typename MatrixType> 

void renumberExternalsAndCount(MatrixType A, typename MatrixType::GlobalOrdinalType start_row, typename MatrixType::GlobalOrdinalType stop_row, typename MatrixType::GlobalOrdinalType *num_externals,
                               sycl::nd_item<3> item_ct1,
                               volatile typename MatrixType::GlobalOrdinalType *red) {
  typedef typename MatrixType::GlobalOrdinalType GlobalOrdinalType;
  GlobalOrdinalType sum=0;
  for (int row_idx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                     item_ct1.get_local_id(2);
       row_idx < A.num_rows; row_idx += item_ct1.get_local_range().get(2) *
                                        item_ct1.get_group_range(2))
  {
    int offset=row_idx;
    for(int j=0;j<A.num_cols_per_row;++j) {
      GlobalOrdinalType col=A.cols[offset];
      if(col==-1) break;
      GlobalOrdinalType new_col;
      //if this column is larger than the number of rows in A it is an external
      if( ( col<start_row || col>stop_row ) ) {
        new_col=-(col + 2);
        sum++;
      } else {
        new_col = col -start_row;
      }
      A.cols[offset] = new_col;

      offset+=A.pitch;
    }
  }
  //TODO reduce

  red[item_ct1.get_local_id(2)] = sum;
  /*
  DPCT1065:51: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier(); if (item_ct1.get_local_id(2) < 256) {
      sum += red[item_ct1.get_local_id(2) + 256];
      red[item_ct1.get_local_id(2)] = sum;
  }
  /*
  DPCT1065:52: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier(); if (item_ct1.get_local_id(2) < 128) {
      sum += red[item_ct1.get_local_id(2) + 128];
      red[item_ct1.get_local_id(2)] = sum;
  }
  /*
  DPCT1065:53: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier(); if (item_ct1.get_local_id(2) < 64) {
      sum += red[item_ct1.get_local_id(2) + 64];
      red[item_ct1.get_local_id(2)] = sum;
  }
  /*
  DPCT1065:54: Consider replacing sycl::nd_item::barrier() with
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

  if (item_ct1.get_local_id(2) == 0)
    /*
    DPCT1039:55: The generated code assumes that "num_externals" points to the
    global memory address space. If it points to a local memory address space,
    replace "dpct::atomic_fetch_add" with "dpct::atomic_fetch_add<typename
    struct PODELLMatrix<double, int, int>::GlobalOrdinalType,
    sycl::access::address_space::local_space>".
    */
    sycl::atomic<
        typename struct PODELLMatrix<double, int, int>::GlobalOrdinalType>(
        sycl::global_ptr<
            typename struct PODELLMatrix<double, int, int>::GlobalOrdinalType>(
            num_externals))
        .fetch_add(sum);
}

template <typename MatrixType> 
void renumberExternals(MatrixType A, 
    typename MatrixType::GlobalOrdinalType *externals_list, 
    typename MatrixType::GlobalOrdinalType num_externals,
    typename MatrixType::GlobalOrdinalType *local_index_map,
    sycl::nd_item<3> item_ct1,
    const sycl::stream &stream_ct1) {
  typedef typename MatrixType::GlobalOrdinalType GlobalOrdinal;

  //copy external_local_index to device
  for (int row_idx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                     item_ct1.get_local_id(2);
       row_idx < A.num_rows; row_idx += item_ct1.get_local_range().get(2) *
                                        item_ct1.get_group_range(2))
  {
    int offset=row_idx;
    for(int j=0;j<A.num_cols_per_row;++j) {
      GlobalOrdinal col=A.cols[offset];
      if(col==-1) break;
      //if external
      if( col<-1 ) {
        //compute global column index
        col=-col -2;
        //find global column index entry
        GlobalOrdinal loc=binarySearch(externals_list,0,num_externals-1,col);
        /*
        DPCT1015:56: Output needs adjustment.
        */
        if (loc == -1) stream_ct1
            << "Error unable to find column: %d, in external list\n";
        assert(loc!=-1);
        //map to new local index
        GlobalOrdinal new_col = local_index_map[loc];
        //actually renumber
        A.cols[offset] = new_col;
      } 
      offset+=A.pitch;
    }
  }
}


//The following function was converted from Mike's HPCCG code.
template<typename MatrixType>
void
make_local_matrix(MatrixType& A)
{
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.default_queue();
#ifdef HAVE_MPI
  int numprocs = 1, myproc = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myproc);

  if (numprocs < 2) {
    A.num_cols = A.rows.size();
    A.has_local_indices = true;
    return;
  }

  typedef typename MatrixType::GlobalOrdinalType GlobalOrdinal;
  typedef typename MatrixType::LocalOrdinalType LocalOrdinal;
  typedef typename MatrixType::ScalarType Scalar;

  std::map<GlobalOrdinal,GlobalOrdinal> externals;
  LocalOrdinal num_external = 0;

  //Extract Matrix pieces

  size_t local_nrow = A.rows.size();
  GlobalOrdinal start_row = local_nrow>0 ? A.rows[0] : -1;
  GlobalOrdinal stop_row  = local_nrow>0 ? A.rows[local_nrow-1] : -1;

  // We need to convert the index values for the rows on this processor
  // to a local index space. We need to:
  // - Determine if each index reaches to a local value or external value
  // - If local, subtract start_row from index value to get local index
  // - If external, find out if it is already accounted for.
  //   - If so, then do nothing,
  //   - otherwise
  //     - add it to the list of external indices,
  //     - find out which processor owns the value.
  //     - Set up communication for sparse MV operation

  ///////////////////////////////////////////
  // Scan the indices and transform to local
  ///////////////////////////////////////////

  //Wait for rows and cols to finish transfer
  CudaManager::e1.wait_and_throw();

  std::vector<GlobalOrdinal>& external_index = A.external_index;
#define CUDA_MAKE_LOCAL
#ifdef CUDA_MAKE_LOCAL
  dpct::device_vector<GlobalOrdinal> num_cols_est(1, 0);
  int BLOCK_SIZE=512;
  int MAX_BLOCKS=8192;
  int NUM_BLOCKS =
      std::min(MAX_BLOCKS, (int)(A.rows.size() + BLOCK_SIZE - 1) / BLOCK_SIZE);

  /*
  DPCT1049:58: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  CudaManager::s1->submit([&](sycl::handler &cgh) {
    sycl::accessor<GlobalOrdinalType, 1, sycl::access_mode::read_write,
                   sycl::access::target::local>
        red_acc_ct1(sycl::range<1>(512), cgh);

    auto A_getPOD_ct0 = A.getPOD();
    auto thrust_raw_pointer_cast_num_cols_est_ct3 =
        dpct::get_raw_pointer(&num_cols_est[0]);

    cgh.parallel_for<dpct_kernel_name<class renumberExternalsAndCount_bbf462,
                                      MatrixType /*Fix the type mannually*/>>(
        sycl::nd_range<3>(sycl::range<3>(1, 1, NUM_BLOCKS) *
                              sycl::range<3>(1, 1, BLOCK_SIZE),
                          sycl::range<3>(1, 1, BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
          renumberExternalsAndCount(A_getPOD_ct0, start_row, stop_row,
                                    thrust_raw_pointer_cast_num_cols_est_ct3,
                                    item_ct1, red_acc_ct1.get_pointer());
        });
  });

  ColMarkMap<GlobalOrdinal> d_map(num_cols_est[0]*20); //TODO tune this multiplier and test
  BLOCK_SIZE=256;
  NUM_BLOCKS = std::min(MAX_BLOCKS, (int)(d_map.table.size() + BLOCK_SIZE - 1) /
                                        BLOCK_SIZE);

  //insert all external columns into an array
  /*
  DPCT1049:59: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  CudaManager::s1->submit([&](sycl::handler &cgh) {
    auto A_getPOD_ct0 = A.getPOD();
    auto d_map_getPOD_ct1 = d_map.getPOD();

    cgh.parallel_for<dpct_kernel_name<class markExternalColumnsInMap_2f8471,
                                      GlobalOrdinal /*Fix the type mannually*/,
                                      Scalar /*Fix the type mannually*/>>(
        sycl::nd_range<3>(sycl::range<3>(1, 1, NUM_BLOCKS) *
                              sycl::range<3>(1, 1, BLOCK_SIZE),
                          sycl::range<3>(1, 1, BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
          markExternalColumnsInMap(A_getPOD_ct0, d_map_getPOD_ct1, item_ct1);
        });
  });
  cudaCheckError();

  //sort the map table
  oneapi::dpl::sort(
      oneapi::dpl::execution::make_device_policy<class Policy_f7321c>(
          dpct::get_default_queue()),
      d_map.table.begin(), d_map.table.end());
  cudaCheckError();
  
  //compute unique elements
  num_external = std::unique(d_map.table.begin(), d_map.table.end()) -
                 d_map.table.begin() - 1;
  cudaCheckError();

  external_index.resize(num_external);
  //printf("external_index.size: %d, d_map.table.size: %d\n",external_index.size(), num_external);
  //copy unique elements external_index (remove first element -1)
  dpct::dpct_memcpy(&external_index[0], dpct::get_raw_pointer(&d_map.table[1]),
                    num_external * sizeof(GlobalOrdinal), dpct::device_to_host);
  cudaCheckError();
#else
  #pragma omp parallel for
  for(size_t i=0; i<A.rows.size(); ++i) {
    int row=A.rows[i];
    int local_row=A.get_local_row(row);
    GlobalOrdinal* Acols = &A.cols[0];
    size_t row_len = 27;

    for(size_t j=0; j<row_len; ++j) {
      int idx=local_row+j*A.pitch;
      GlobalOrdinal cur_ind = Acols[idx];
      if(cur_ind==-1) break;

      if (start_row <= cur_ind && cur_ind <= stop_row) {
        Acols[idx] -= start_row;
      }
      else { // Must find out if we have already set up this point

#pragma omp critical 
{
        if (externals.find(cur_ind) == externals.end()) {
          externals[cur_ind] = num_external++;
          external_index.push_back(cur_ind);
        }
}
        // Mark index as external by adding 2 and negating it  (-1 is reserved to mark padding)
        Acols[idx] = -(Acols[idx] + 2);
      }
    }
  }
#endif

  ////////////////////////////////////////////////////////////////////////
  // Go through list of externals to find out which processors must be accessed.
  ////////////////////////////////////////////////////////////////////////

  std::vector<GlobalOrdinal> tmp_buffer(numprocs, 0); // Temp buffer space needed below

  // Build list of global index offset

  std::vector<GlobalOrdinal> global_index_offsets(numprocs, 0);

  tmp_buffer[myproc] = start_row; // This is my start row

  // This call sends the start_row of each ith processor to the ith
  // entry of global_index_offsets on all processors.
  // Thus, each processor knows the range of indices owned by all
  // other processors.
  // Note: There might be a better algorithm for doing this, but this
  //       will work...

  nvtxRangeId_t r0=nvtxRangeStartA("MPI Reduce");
  MPI_Datatype mpi_dtype = TypeTraits<GlobalOrdinal>::mpi_type();
  MPI_Allreduce(&tmp_buffer[0], &global_index_offsets[0], numprocs, mpi_dtype,
                MPI_SUM, MPI_COMM_WORLD);
  nvtxRangeEnd(r0);

  // Go through list of externals and find the processor that owns each
  std::vector<int> external_processor(num_external);
  
  nvtxRangeId_t r00=nvtxRangeStartA("Set External Processors");
  #pragma omp parallel for
  for(LocalOrdinal i=0; i<num_external; ++i) {
    GlobalOrdinal cur_ind = external_index[i];
    for(int j=numprocs-1; j>=0; --j) {
      if (global_index_offsets[j] <= cur_ind && global_index_offsets[j] >= 0) {
        external_processor[i] = j;
        break;
      }
    }
  }
  nvtxRangeEnd(r00);

  /////////////////////////////////////////////////////////////////////////
  // Sift through the external elements. For each newly encountered external
  // point assign it the next index in the sequence. Then look for other
  // external elements who are updated by the same node and assign them the next
  // set of index numbers in the sequence (ie. elements updated by the same node
  // have consecutive indices).
  /////////////////////////////////////////////////////////////////////////

  nvtxRangeId_t r01=nvtxRangeStartA("Create external_local_index");
  size_t count = local_nrow;
  std::vector<GlobalOrdinal>& external_local_index = A.external_local_index;
  external_local_index.assign(num_external, -1);

  for(LocalOrdinal i=0; i<num_external; ++i) {
    if (external_local_index[i] == -1) {
      external_local_index[i] = count++;

      for(LocalOrdinal j=i+1; j<num_external; ++j) {
        if (external_processor[j] == external_processor[i])
          external_local_index[j] = count++;
      }
    }
  }
  nvtxRangeEnd(r01);
#ifdef CUDA_MAKE_LOCAL
  //copy external_local_index to device
  dpct::device_vector<GlobalOrdinal> d_external_local_index;
  d_external_local_index.assign(external_local_index.begin(),external_local_index.end());
  cudaCheckError();

  //renumber externals list
  BLOCK_SIZE=256;
  MAX_BLOCKS=32768;
  NUM_BLOCKS =
      std::min(MAX_BLOCKS, (int)(A.rows.size() + BLOCK_SIZE - 1) / BLOCK_SIZE);
  /*
  DPCT1049:60: The workgroup size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the workgroup size if needed.
  */
  dpct::get_default_queue().submit([&](sycl::handler &cgh) {
    sycl::stream stream_ct1(64 * 1024, 80, cgh);

    auto A_getPOD_ct0 = A.getPOD();
    auto thrust_raw_pointer_cast_d_map_table_ct1 =
        dpct::get_raw_pointer(&d_map.table[1]);
    auto thrust_raw_pointer_cast_d_external_local_index_ct3 =
        dpct::get_raw_pointer(&d_external_local_index[0]);

    cgh.parallel_for<dpct_kernel_name<class renumberExternals_8533eb,
                                      MatrixType /*Fix the type mannually*/>>(
        sycl::nd_range<3>(sycl::range<3>(1, 1, NUM_BLOCKS) *
                              sycl::range<3>(1, 1, BLOCK_SIZE),
                          sycl::range<3>(1, 1, BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
          renumberExternals(
              A_getPOD_ct0, thrust_raw_pointer_cast_d_map_table_ct1,
              num_external, thrust_raw_pointer_cast_d_external_local_index_ct3,
              item_ct1, stream_ct1);
        });
  });
  cudaCheckError();

#else

  //TODO time consuming
  //TODO, This only updates Acols.  We could move externals and externel_local_index to the device and then do this there...
  #pragma omp parallel for
  for(size_t i=0; i<local_nrow; ++i) {
    int row=A.rows[i]; 
    int local_row=A.get_local_row(row);
    size_t row_len = 27;
    GlobalOrdinal* Acols = &A.cols[0];

    for(size_t j=0; j<row_len; ++j) {
      int idx=local_row+j*A.pitch;
      int cur_ind=A.cols[idx];
      if(cur_ind==-1) break;
      if (Acols[idx] < 0) { // Change index values of externals
        GlobalOrdinal cur_ind = -Acols[idx] - 2;
        Acols[idx] = external_local_index[externals[cur_ind]];
      }
    }
  }
#endif

  std::vector<int> new_external_processor(num_external, 0);

  //TODO move to device
  nvtxRangeId_t r1=nvtxRangeStartA("assign new external");
  #pragma omp parallel for
  for(int i=0; i<num_external; ++i) {
    new_external_processor[external_local_index[i]-local_nrow] =
      external_processor[i];
  }
  nvtxRangeEnd(r1);

  ////////////////////////////////////////////////////////////////////////
  ///
  // Count the number of neighbors from which we receive information to update
  // our external elements. Additionally, fill the array tmp_neighbors in the
  // following way:
  //      tmp_neighbors[i] = 0   ==>  No external elements are updated by
  //                              processor i.
  //      tmp_neighbors[i] = x   ==>  (x-1)/numprocs elements are updated from
  //                              processor i.
  ///
  ////////////////////////////////////////////////////////////////////////

  std::vector<GlobalOrdinal> tmp_neighbors(numprocs, 0);

  int num_recv_neighbors = 0;
  int length             = 1;

  nvtxRangeId_t r2=nvtxRangeStartA("create tmp_neighbors");
  for(LocalOrdinal i=0; i<num_external; ++i) {
    if (tmp_neighbors[new_external_processor[i]] == 0) {
      ++num_recv_neighbors;
      tmp_neighbors[new_external_processor[i]] = 1;
    }
    tmp_neighbors[new_external_processor[i]] += numprocs;
  }
  nvtxRangeEnd(r2);


  /// sum over all processor all the tmp_neighbors arrays ///

  nvtxRangeId_t r20=nvtxRangeStartA("MPI All Reduce");
  MPI_Allreduce(&tmp_neighbors[0], &tmp_buffer[0], numprocs, mpi_dtype,
                MPI_SUM, MPI_COMM_WORLD);
  nvtxRangeEnd(r20);

  // decode the combined 'tmp_neighbors' (stored in tmp_buffer)
  // array from all the processors

  GlobalOrdinal num_send_neighbors = tmp_buffer[myproc] % numprocs;

  /// decode 'tmp_buffer[myproc] to deduce total number of elements
  //  we must send

  GlobalOrdinal total_to_be_sent = (tmp_buffer[myproc] - num_send_neighbors) / numprocs;

  ///////////////////////////////////////////////////////////////////////
  ///
  // Make a list of the neighbors that will send information to update our
  // external elements (in the order that we will receive this information).
  ///
  ///////////////////////////////////////////////////////////////////////
  
  nvtxRangeId_t r3=nvtxRangeStartA("create recv list");
  std::vector<int> recv_list;
  recv_list.push_back(new_external_processor[0]);
  for(LocalOrdinal i=1; i<num_external; ++i) {
    if (new_external_processor[i-1] != new_external_processor[i]) {
      recv_list.push_back(new_external_processor[i]);
    }
  }
  nvtxRangeEnd(r3);

  //
  // Send a 0 length message to each of our recv neighbors
  //

  std::vector<int> send_list(num_send_neighbors, 0);

  //
  // first post receives, these are immediate receives
  // Do not wait for result to come, will do that at the
  // wait call below.
  //
  int MPI_MY_TAG = 99;

  nvtxRangeId_t r4=nvtxRangeStartA("MPI Communication");
  std::vector<MPI_Request> request(num_send_neighbors);
  for(int i=0; i<num_send_neighbors; ++i) {
    MPI_Irecv(&tmp_buffer[i], 1, mpi_dtype, MPI_ANY_SOURCE, MPI_MY_TAG,
              MPI_COMM_WORLD, &request[i]);
  }

  // send messages

  for(int i=0; i<num_recv_neighbors; ++i) {
    MPI_Send(&tmp_buffer[i], 1, mpi_dtype, recv_list[i], MPI_MY_TAG,
             MPI_COMM_WORLD);
  }

  ///
  // Receive message from each send neighbor to construct 'send_list'.
  ///

  MPI_Status status;
  for(int i=0; i<num_send_neighbors; ++i) {
    if (MPI_Wait(&request[i], &status) != MPI_SUCCESS) {
      std::cerr << "MPI_Wait error\n"<<std::endl;
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
    send_list[i] = status.MPI_SOURCE;
  }
  nvtxRangeEnd(r4);

  //////////////////////////////////////////////////////////////////////
  ///
  // Compare the two lists. In most cases they should be the same.
  // However, if they are not then add new entries to the recv list
  // that are in the send list (but not already in the recv list).
  ///
  //////////////////////////////////////////////////////////////////////

  nvtxRangeId_t r5=nvtxRangeStartA("add to recv_list");
  for(int j=0; j<num_send_neighbors; ++j) {
    int found = 0;
    for(int i=0; i<num_recv_neighbors; ++i) {
      if (recv_list[i] == send_list[j]) found = 1;
    }

    if (found == 0) {
      recv_list.push_back(send_list[j]);
      ++num_recv_neighbors;
    }
  }
  nvtxRangeEnd(r5);

  nvtxRangeId_t r50=nvtxRangeStartA("allocation");
  num_send_neighbors = num_recv_neighbors;
  request.resize(num_send_neighbors);

  A.elements_to_send.assign(total_to_be_sent, 0);
  A.d_elements_to_send.resize(total_to_be_sent);

#ifndef GPUDIRECT
  A.send_buffer.resize(total_to_be_sent);
#endif
  A.d_send_buffer.resize(total_to_be_sent);
  //
  // Create 'new_external' which explicitly put the external elements in the
  // order given by 'external_local_index'
  //

  std::vector<GlobalOrdinal> new_external(num_external);
  nvtxRangeEnd(r50);


  nvtxRangeId_t r51=nvtxRangeStartA("assign new external");
  #pragma omp_parallel for
  for(LocalOrdinal i=0; i<num_external; ++i) {
    new_external[external_local_index[i] - local_nrow] = external_index[i];
  }
  nvtxRangeEnd(r51);

  /////////////////////////////////////////////////////////////////////////
  //
  // Send each processor the global index list of the external elements in the
  // order that I will want to receive them when updating my external elements.
  //
  /////////////////////////////////////////////////////////////////////////

  std::vector<int> lengths(num_recv_neighbors);

  ++MPI_MY_TAG;

  // First post receives

  nvtxRangeId_t r52=nvtxRangeStartA("MPI Communication");
  for(int i=0; i<num_recv_neighbors; ++i) {
    int partner = recv_list[i];
    MPI_Irecv(&lengths[i], 1, MPI_INT, partner, MPI_MY_TAG, MPI_COMM_WORLD,
              &request[i]);
  }

  std::vector<int>& neighbors = A.neighbors;
  std::vector<int>& recv_length = A.recv_length;
  std::vector<int>& send_length = A.send_length;

  neighbors.resize(num_recv_neighbors, 0);
  A.request.resize(num_recv_neighbors);
  recv_length.resize(num_recv_neighbors, 0);
  send_length.resize(num_recv_neighbors, 0);

  LocalOrdinal j = 0;
  for(int i=0; i<num_recv_neighbors; ++i) {
    int start = j;
    int newlength = 0;

    //go through list of external elements until updating
    //processor changes

    while((j < num_external) &&
          (new_external_processor[j] == recv_list[i])) {
      ++newlength;
      ++j;
      if (j == num_external) break;
    }

    recv_length[i] = newlength;
    neighbors[i] = recv_list[i];

    length = j - start;
    MPI_Send(&length, 1, MPI_INT, recv_list[i], MPI_MY_TAG, MPI_COMM_WORLD);
  }

  // Complete the receives of the number of externals

  for(int i=0; i<num_recv_neighbors; ++i) {
    if (MPI_Wait(&request[i], &status) != MPI_SUCCESS) {
      std::cerr << "MPI_Wait error\n"<<std::endl;
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
    send_length[i] = lengths[i];
  }

  ////////////////////////////////////////////////////////////////////////
  // Build "elements_to_send" list. These are the x elements I own
  // that need to be sent to other processors.
  ////////////////////////////////////////////////////////////////////////

  ++MPI_MY_TAG;

  j = 0;
  for(int i=0; i<num_recv_neighbors; ++i) {
    MPI_Irecv(&A.elements_to_send[j], send_length[i], mpi_dtype, neighbors[i],
              MPI_MY_TAG, MPI_COMM_WORLD, &request[i]);
    j += send_length[i];
  }

  j = 0;
  for(int i=0; i<num_recv_neighbors; ++i) {
    LocalOrdinal start = j;
    LocalOrdinal newlength = 0;

    // Go through list of external elements
    // until updating processor changes. This is redundant, but
    // saves us from recording this information.

    while((j < num_external) &&
          (new_external_processor[j] == recv_list[i])) {
      ++newlength;
      ++j;
      if (j == num_external) break;
    }
    MPI_Send(&new_external[start], j-start, mpi_dtype, recv_list[i],
             MPI_MY_TAG, MPI_COMM_WORLD);
  }

  // receive from each neighbor the global index list of external elements

  for(int i=0; i<num_recv_neighbors; ++i) {
    if (MPI_Wait(&request[i], &status) != MPI_SUCCESS) {
      std::cerr << "MPI_Wait error\n"<<std::endl;
      MPI_Abort(MPI_COMM_WORLD, -1);
    }
  }
  nvtxRangeEnd(r52);

  /// replace global indices by local indices ///

  //TODO move this to the device (trivial)
  #pragma omp parallel for
  for(GlobalOrdinal i=0; i<total_to_be_sent; ++i) {
    A.elements_to_send[i] -= start_row;
  }

  dpct::async_dpct_memcpy(dpct::get_raw_pointer(&A.d_elements_to_send[0]),
                          &A.elements_to_send[0],
                          sizeof(GlobalOrdinal) * A.elements_to_send.size(),
                          dpct::host_to_device, *CudaManager::s1);
  cudaCheckError();
  /*
  DPCT1012:57: Detected kernel execution time measurement pattern and generated
  an initial code for time measurements in SYCL. You can change the way time is
  measured depending on your goals.
  */
  e1_ct1 = std::chrono::steady_clock::now();
  CudaManager::e1 = CudaManager::s1->submit_barrier();
  //////////////////
  // Finish up !!
  //////////////////
  A.num_cols = local_nrow + num_external;

#else
  A.num_cols = A.rows.size();
#endif

  A.has_local_indices = true;

#ifdef HAVE_MPI
#ifndef CUDA_MAKE_LOCAL
  cudaMemcpyAsync(thrust::raw_pointer_cast(&A.d_cols[0]), &A.cols[0], sizeof(GlobalOrdinal)*A.cols.size(),cudaMemcpyHostToDevice,CudaManager::s1);
  cudaCheckError();
#endif

#ifdef MATVEC_OVERLAP
  {
    //create external/interal mapping vector for overlap
    const int BLOCK_SIZE=256;
    const int MAX_BLOCKS=32768;

    const int NUM_BLOCKS = std::min(
        MAX_BLOCKS, (int)(A.rows.size() + BLOCK_SIZE - 1) / BLOCK_SIZE);
    CudaManager::s1->submit([&](sycl::handler &cgh) {
      auto A_getPOD_ct0 = A.getPOD();

      cgh.parallel_for<
          dpct_kernel_name<class createExternalMapping_2ca3f7,
                           PlaceHolder /*Fix the type mannually*/>>(
          sycl::nd_range<3>(sycl::range<3>(1, 1, NUM_BLOCKS) *
                                sycl::range<3>(1, 1, BLOCK_SIZE),
                            sycl::range<3>(1, 1, BLOCK_SIZE)),
          [=](sycl::nd_item<3> item_ct1) {
            createExternalMapping(A_getPOD_ct0, item_ct1);
          });
    });
    cudaCheckError();
  }
#endif
#endif

}

}//namespace miniFE

#endif

