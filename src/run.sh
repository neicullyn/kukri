export PATH=$PATH:/usr/local/cuda-6.5/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-6.5/lib64

make -f Makefile

mkdir -p log

name=$(date +log/%Y%m%d_%H%M%S.txt)
./kukri_test 1000 1000 1000 | tee $name