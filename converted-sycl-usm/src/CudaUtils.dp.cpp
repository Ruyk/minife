#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include <CudaUtils.h>

namespace miniFE {

  sycl::queue *CudaManager::s1;
  sycl::queue *CudaManager::s2;
  sycl::event CudaManager::e1;
  sycl::event CudaManager::e2;
  bool CudaManager::initialized=false;

}
