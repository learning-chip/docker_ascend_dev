# Docker environment for operator kernel development on Ascend NPU

## Build image

Prepare installer (example for 910B)

```bash
cd installers
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.2.RC2/Ascend-cann-toolkit_8.2.RC2_linux-x86_64.run
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.2.RC2/Ascend-cann-kernels-910b_8.2.RC2_linux-x86_64.run
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.2.RC2/Ascend-cann-nnal_8.2.RC2_linux-x86_64.run
# could also download during docker build, but too big, better download manually
```

(link obtained from https://www.hiascend.com/developer/download/community/result?module=pt+cann&pt=7.1.0&cann=8.2.RC2)

Put installer under `./installers` dir so that Dockerfile can find it.

Then build image:

```bash
# under top dir
sudo docker build . -t torch_npu_cann:8.2.RC2 \
     -f dockerfiles.x86/dockerfile.torch_npu_cann
```


## Launch docker runtime

```bash
# NOTE: change `--device` to actual device id on your server
sudo docker run --rm -it --ipc=host --privileged \
    --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc  \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v $HOME:/mounted_home \
    -w /mounted_home \
    torch_npu_cann:8.2.RC2 \
    /bin/bash

# inside container
source $CONDA_HOME/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
```

### Run tilelang-ascend examples

```bash
# inside container launched above
cd /installers/tilelang-ascend/
source set_env.sh
cd examples/gemm
python example_gemm.py  # should get "Kernel Output Match!"
```

### Run catlass examples

```bash
# inside container launched above

# C++ examples
cd /installers/catlass/build/bin
./00_basic_matmul 256 512 1024 0  # should get "Compare success."
./06_optimized_matmul 256 512 1024 0
./12_quant_matmul 256 512 1024 0

python ../../examples/19_mla/gen_data.py 1 1 128 16 16 128 half
./19_mla 1 1 128 16 16 128 --dtype half --datapath ../../examples/19_mla/data --device 0

# Torch examples
cd /installers/catlass/examples/python_extension/tests
pytest -v test_python_extension.py -k "test_basic_matmul"
```


Should get:

```
test_python_extension.py::CatlassTest::test_basic_matmul_pybind PASSED                                                   [ 33%]
test_python_extension.py::CatlassTest::test_basic_matmul_pybind_bf16 PASSED                                              [ 66%]
test_python_extension.py::CatlassTest::test_basic_matmul_torch_lib PASSED                                                [100%]

=============================================== 3 passed, 3 deselected in 6.05s ================================================
```
