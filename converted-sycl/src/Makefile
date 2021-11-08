#-----------------------------------------------------------------------

MINIFE_TYPES =  \
        -DMINIFE_SCALAR=double   \
        -DMINIFE_LOCAL_ORDINAL=int      \
        -DMINIFE_GLOBAL_ORDINAL=int

MINIFE_MATRIX_TYPE = -DMINIFE_ELL_MATRIX

#-----------------------------------------------------------------------

CFLAGS = -O3
CXXFLAGS = -O3

# For debugging, the macro MINIFE_DEBUG will cause miniFE to dump a log file
# from each proc containing various information.
# This macro will also enable a somewhat expensive range-check on indices in
# the exchange_externals function.

#CPPFLAGS = -I. -I../utils -I../fem $(MINIFE_TYPES) $(MINIFE_MATRIX_TYPE) -DHAVE_MPI -DMPICH_IGNORE_CXX_SEEK  #-DMINIFE_DEBUG
CPPFLAGS = -I. -I../utils -I../fem $(MINIFE_TYPES) $(MINIFE_MATRIX_TYPE) -DHAVE_MPI -DMPICH_IGNORE_CXX_SEEK -DMATVEC_OVERLAP #-DGPUDIRECT #-DMINIFE_DEBUG
#CPPFLAGS = -I. -I../utils -I../fem $(MINIFE_TYPES) $(MINIFE_MATRIX_TYPE) #-DHAVE_MPI -DMPICH_IGNORE_CXX_SEEK -DMATVEC_OVERLAP #-DGPUDIRECT #-DMATVEC_OVERLAP #-DMINIFE_DEBUG
#CPPFLAGS = -I. -I../utils -I../fem $(MINIFE_TYPES) $(MINIFE_MATRIX_TYPE) -DHAVE_MPI -DMPICH_IGNORE_CXX_SEEK -DGPUDIRECT -DMATVEC_OVERLAP #-DMINIFE_DEBUG

LDFLAGS=  
LIBS= -l nvToolsExt 

# The MPICH_IGNORE_CXX_SEEK macro is required for some mpich versions,
# such as the one on my cygwin machine.

#mvapich
#MPICFLAGS=-I/usr/local/mvapich/include -pthread 
#MPILDFLAGS=-L/usr/local/mvapich/lib -lmpichcxx -lmpich -lopa -lmpl -lcudart -lcuda -libverbs -ldl -lrt -lm -lpthread 
#c++ -L/usr/local/cuda//lib64 -L/usr/local/cuda//lib -L/lib -L/lib -L/lib -Wl,-rpath,/lib -L/lib -Wl,-rpath,/lib -L/lib -Wl,-rpath,/lib -L/lib -L/lib -I/usr/local/mvapich/include -L/usr/local/mvapich/lib -lmpichcxx -lmpich -lopa -lmpl -lcudart -lcuda -libverbs -ldl -lrt -lm -lpthread

#openmpi
# -I$(TBB_HOME)/include/ 
MPICFLAGS=-I$(MPI_HOME)/include -I$(DPL_HOME)/include/ -I$(DPCPP_HOME)/include/sycl/ -I$(DPCPP_HOME)/compiler/include/ -I$(DPCT_HOME)/include/ -pthread  -std=c++17 -D_GLIBCXX_USE_TBB_PAR_BACKEND=0 -DPSTL_USE_PARALLEL_POLICIES=0 -DONEDPL_USE_DPCPP_BACKEND=1 -DONEDPL_USE_PREDEFINED_POLICIES=1 -DONEDPL_USE_DEFAULT_POLICIES=1
MPILDFLAGS=-L$(MPI_HOME)/lib -L$(DPCPP_HOME)/lib/ -lsycl -lmpi -lmpicxx -ldl -lm -lrt -lnsl -lutil -lm -ldl -std=c++17

NVCCFLAGS=

NVCC=dpcpp -fsycl -fopenmp -std=c++17 -D_GLIBCXX_USE_TBB_PAR_BACKEND=0 -DONEDPL_USE_DEFAULT_POLICIES=1 -DONEDPL_USE_DPCPP_BACKEND=1 -DONEDPL_USE_PREDEFINED_POLICIES=1 -g
CXX=mpicxx  -fPIE -fopenmp -g
CC=mpiccc  -fPIE -g

include make_targets
