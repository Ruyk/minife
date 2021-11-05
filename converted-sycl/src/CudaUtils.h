
#ifndef _H_MINIFE_CUDA_UTILS
#define _H_MINIFE_CUDA_UTILS

#define DPCT_USM_LEVEL_NONE
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <assert.h>
#include <shfl.h>
#include <device_atomic_functions.h>
#include <cmath>

#include <algorithm>

__inline__ double miniFEAtomicAdd(double* address, double val)
{
    unsigned long long int *address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        /*
        DPCT1039:0: The generated code assumes that "address_as_ull" points to
        the global memory address space. If it points to a local memory address
        space, replace "dpct::atomic_compare_exchange_strong" with
        "dpct::atomic_compare_exchange_strong<unsigned long long,
        sycl::access::address_space::local_space>".
        */
        old = dpct::atomic_compare_exchange_strong(
            address_as_ull, assumed,
            (unsigned long long)(sycl::bit_cast<long long>(
                val + sycl::bit_cast<double>(assumed))));
    } while (assumed != old);
    return sycl::bit_cast<double>(old);
}

namespace miniFE {

/*
DPCT1010:9: SYCL uses exceptions to report errors and does not use the error
codes. The call was replaced with 0. You need to rewrite this code.
*/
/*
DPCT1009:10: SYCL uses exceptions to report errors and does not use the error
codes. The original code was commented out and a warning string was inserted.
You need to rewrite this code.
*/
#define cudaCheckError()                                                       \
    { int e = 0; }

#if defined(DPCT_COMPATIBILITY_TEMP) & (DPCT_COMPATIBILITY_TEMP < 350)
template <class T> static __device__ inline T __ldg(T* ptr) { return *ptr; }
#endif

template <typename ValueType> 
  struct convert {
    union {
      ValueType v;
      int i;
    };
  };
template<typename ValueType> 
__inline__
ValueType __compare_and_swap_xor(ValueType val, int mask,
                                 sycl::nd_item<3> item_ct1, int ASCENDING=true) {
  int laneId =
      item_ct1.get_local_id(2) % 32; // is there a better way to get this?
  int src=mask^laneId;

  convert<ValueType> new_val;
  new_val.v=val;
  new_val.i=__shfl(new_val.i,src);

  return (ASCENDING ^ (laneId<src) ^ (val<new_val.v)) ? val : new_val.v;
}


template<typename ValueType> 
__inline__
ValueType __sort(ValueType val, sycl::nd_item<3> item_ct1, int ASCENDING=true) {
  int laneId = item_ct1.get_local_id(2) % 32;
  int DIRECTION=ASCENDING^(laneId/2%2);
  val = __compare_and_swap_xor(val, 1, item_ct1, DIRECTION);
  DIRECTION=ASCENDING^(laneId/4%2);
  val = __compare_and_swap_xor(val, 2, item_ct1, DIRECTION);
  val = __compare_and_swap_xor(val, 1, item_ct1, DIRECTION);
  DIRECTION=ASCENDING^(laneId/8%2);
  val = __compare_and_swap_xor(val, 4, item_ct1, DIRECTION);
  val = __compare_and_swap_xor(val, 2, item_ct1, DIRECTION);
  val = __compare_and_swap_xor(val, 1, item_ct1, DIRECTION);
  DIRECTION=ASCENDING^(laneId/16%2);
  val = __compare_and_swap_xor(val, 8, item_ct1, DIRECTION);
  val = __compare_and_swap_xor(val, 4, item_ct1, DIRECTION);
  val = __compare_and_swap_xor(val, 2, item_ct1, DIRECTION);
  val = __compare_and_swap_xor(val, 1, item_ct1, DIRECTION);
  DIRECTION=ASCENDING;
  val = __compare_and_swap_xor(val, 16, item_ct1, DIRECTION);
  val = __compare_and_swap_xor(val, 8, item_ct1, DIRECTION);
  val = __compare_and_swap_xor(val, 4, item_ct1, DIRECTION);
  val = __compare_and_swap_xor(val, 2, item_ct1, DIRECTION);
  val = __compare_and_swap_xor(val, 1, item_ct1, DIRECTION);

  return val;
}

template<typename GlobalOrdinal>
  __inline__
  GlobalOrdinal lowerBound(const GlobalOrdinal *ptr, GlobalOrdinal low, GlobalOrdinal high, const GlobalOrdinal val) {

  //printf("Binary Search  for %d\n",val);
  
  while(high>=low)
  {
    GlobalOrdinal mid=low+(high-low)/2;
    GlobalOrdinal mval;
    /*
    DPCT1026:6: The call to __ldg was removed because there is no correspoinding
    API in DPC++.
    */
    mval = *(ptr + mid);
    //printf("low: %d, high: %d, mid: %d, val: %d\n",low,high,mid, mval);
    if(mval>val)
      high=mid-1;
    else if (mval<val)
      low=mid+1;
    else
    {
      //printf("Found %d at index: %d\n", val, mid);
      return mid;
    }
  }

  /*
  DPCT1026:7: The call to __ldg was removed because there is no correspoinding
  API in DPC++.
  */
  if (*(ptr + high) < val) {
    //printf(" not found returning %d, (%d,%d)\n",high,low,high);
    return high;
  }
  else {
    //printf(" not found returning %d, (%d,%d)\n",high-1,low,high);
    return high-1;
  }
}

template<typename GlobalOrdinal>
  __inline__
  GlobalOrdinal binarySearch(const GlobalOrdinal *ptr, GlobalOrdinal low_, GlobalOrdinal high_, const GlobalOrdinal val) {

    GlobalOrdinal low=low_;
    GlobalOrdinal high=high_;

    //printf("%d:%d, Binary Search  for %d, low: %d, high: %d\n",threadIdx.x,val,val,low,high);

    while(high>=low)
    {
      GlobalOrdinal mid=low+(high-low)/2;
      GlobalOrdinal mval;
      mval=ptr[mid];
      //TODO: use ldg
      //mval=__ldg(ptr+mid);
      //printf("%d:%d, low: %d, high: %d, mid: %d, val: %d\n",threadIdx.x,val,low,high,mid, mval);
      if(mval>val)
        high=mid-1;
      else if (mval<val)
        low=mid+1;
      else
      {
       // printf("%d:%d, Found %d at index: %d\n", threadIdx.x,val, val, mid);
        return mid;
      }
    }
    //printf("%d,%d, not found\n",threadIdx.x,val);
    //not found
    return -1;  
}

class CudaManager {
  public:
    static void initialize() {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
      if(!initialized) {
        s1 = dpct::get_current_device().create_queue();
        s2 = dpct::get_current_device().create_queue();
        /*
        DPCT1026:1: The call to cudaEventCreateWithFlags was removed because
        this call is redundant in DPC++.
        */
        /*
        DPCT1026:2: The call to cudaEventCreateWithFlags was removed because
        this call is redundant in DPC++.
        */
        initialized = true;
      }
    };
    static void finalize() {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
      if(initialized) {
        /*
        DPCT1026:3: The call to cudaEventDestroy was removed because this call
        is redundant in DPC++.
        */
        /*
        DPCT1026:4: The call to cudaEventDestroy was removed because this call
        is redundant in DPC++.
        */
        dpct::get_current_device().destroy_queue(s1);
        dpct::get_current_device().destroy_queue(s2);
        initialized=false;
      }
    };
    static sycl::queue *s1;
    static sycl::queue *s2;
    static sycl::event e1;
    std::chrono::time_point<std::chrono::steady_clock> e1_ct1;
    static sycl::event e2;
    std::chrono::time_point<std::chrono::steady_clock> e2_ct1;

  private:
    static bool initialized;

};

template<class T> 
void cudaMemset_kernel(T* mem, T val, int N, sycl::nd_item<3> item_ct1) {
  for (int idx = item_ct1.get_group(2) * item_ct1.get_local_range().get(2) +
                 item_ct1.get_local_id(2);
       idx < N;
       idx += item_ct1.get_local_range().get(2) * item_ct1.get_group_range(2)) {
    mem[idx]=val;
  }
}

template <class T>
__inline__ void cudaMemset_custom(T *mem, const T val, int N, sycl::queue *s) {
 int BLOCK_SIZE=512;
 int NUM_BLOCKS = std::min(8192, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
  /*
  DPCT1049:5: The workgroup size passed to the SYCL kernel may exceed the limit.
  To get the device limit, query info::device::max_work_group_size. Adjust the
  workgroup size if needed.
  */
  s->submit([&](sycl::handler &cgh) {
    dpct::access_wrapper<T *> mem_acc_ct0(mem, cgh);

    cgh.parallel_for<dpct_kernel_name<class cudaMemset_kernel_10efd5, T>>(
        sycl::nd_range<3>(sycl::range<3>(1, 1, NUM_BLOCKS) *
                              sycl::range<3>(1, 1, BLOCK_SIZE),
                          sycl::range<3>(1, 1, BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
          cudaMemset_kernel(mem_acc_ct0.get_raw_pointer(), val, N, item_ct1);
        });
  });
}

template<int Mark> 
void Marker_kernel() {}

template<int Mark>
void Marker() {
  dpct::get_default_queue()
      .parallel_for<dpct_kernel_name<class Marker_kernel_7f6c28,
                                     dpct_kernel_scalar<Mark>>>(
          sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
          [=](sycl::nd_item<3> item_ct1) {
            Marker_kernel<Mark>();
          });
}

}

#endif
