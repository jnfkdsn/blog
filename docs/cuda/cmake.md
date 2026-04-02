# CMake 构建实践

> 这是我学习用 CMake 管理 CUDA 项目过程中的笔记，记录实际操作和踩坑经历。

## 为什么要学 CMake

单文件用 `nvcc -o softmax softmax.cu` 就行，但真实项目（vLLM、FlashAttention）全部用 CMake 管理几十上百个源文件。看不懂 CMakeLists.txt 就没法在本地编译和修改这些项目。

## 最小 CUDA 项目的 CMakeLists.txt

我的第一个 CMake + CUDA 项目只需要这几行：

```cmake
cmake_minimum_required(VERSION 3.18)
project(hello_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(main main.cpp vector_add.cu)

set_target_properties(main PROPERTIES
    CUDA_ARCHITECTURES "86"
)
```

关键点：`project()` 里加上 `CUDA`，CMake 就会自动用 nvcc 编译 `.cu` 文件、用 g++ 编译 `.cpp` 文件。

## 踩坑记录

### CUDA_ARCHITECTURES 必须设对

一开始没设这个参数，结果运行时报 "no kernel image for device"。

原因：`CUDA_ARCHITECTURES` 决定 nvcc 生成什么架构的 GPU 机器码。我的 GPU 是 RTX 3090（sm_86），如果不设或者设错了就会出问题。

查自己 GPU 的架构：`nvidia-smi` 看 GPU 型号，然后对照表查 Compute Capability。

### find_package 找不到 PyTorch

报错：`Could not find a package configuration file provided by "Torch"`

解决：80% 的 find_package 失败都是 `CMAKE_PREFIX_PATH` 没设对。用这条命令拿到正确路径：

```bash
python -c "import torch; print(torch.utils.cmake_prefix_path)"
```

然后在 cmake 命令里指定：

```bash
cmake -B build -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
```

## 还没搞懂的

- Generator Expression 语法看着很绕，需要更多实践
- CUDA_SEPARABLE_COMPILATION 什么时候真正需要开启
