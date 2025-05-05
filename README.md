# Docker environment for operator kernel development on Ascend NPU

## Build image

Prepare installer (example for 910B)

```bash
cd installers
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.0/Ascend-cann-toolkit_8.0.0_linux-x86_64.run
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.0.0/Ascend-cann-kernels-910b_8.0.0_linux-x86_64.run
# could also download during docker build, but too big, better download manually
```

(link obtained from https://www.hiascend.com/developer/download/community/result?module=pt+cann&cann=8.0.0.beta1)

Put installer under `./installers` dir so that Dockerfile can find it.

Then build image:

```
sudo docker build . -t torch_npu_cann:8.0.beta1 \
     -f dockerfiles.x86/dockerfile.torch_npu_cann
```

## Test run

Run built-in examples

```bash
sudo docker run --rm -it --ipc=host --privileged \
    --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc  \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    torch_npu_cann:8.0.beta1 \
    /bin/bash

# inside container
source $CONDA_HOME/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

# C++ examples
cd build/bin
./00_basic_matmul 256 512 1024 0  # should get "Compare success."
./06_optimized_matmul 256 512 1024 0
./12_quant_matmul 256 512 1024 0

python ../../examples/19_mla/gen_data.py 1 1 128 16 16 128 half
./19_mla 1 1 128 16 16 128 --dtype half --datapath ../../examples/19_mla/data --device 0

# Torch examples
cd /installers/ascendc-templates/examples/python_extension/tests
python test_python_extension.py -v
```


Should get:

```
test_basic_matmul_pybind (__main__.ActTest.test_basic_matmul_pybind) ... ok
test_basic_matmul_pybind_bf16 (__main__.ActTest.test_basic_matmul_pybind_bf16) ... ok
test_basic_matmul_torch_lib (__main__.ActTest.test_basic_matmul_torch_lib) ... ok
test_grouped_matmul_slice_k_pybind (__main__.ActTest.test_grouped_matmul_slice_k_pybind) ... ok
test_grouped_matmul_slice_m_pybind (__main__.ActTest.test_grouped_matmul_slice_m_pybind) ... ok
test_optimized_matmul_pybind (__main__.ActTest.test_optimized_matmul_pybind) ... ok

----------------------------------------------------------------------
Ran 6 tests in 2.402s

OK
```
