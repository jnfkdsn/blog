---
order: 2
title: CMake 构建实践
updated: 2026-05-18
tags: [cmake, cuda, pybind11, torch-extension]
status: draft
---

# CMake 构建实践

## python 调用 cuda 算子

一个最小的 Python CUDA 扩展通常会拆成三层：

```
my_ext/
├── CMakeLists.txt         
├── csrc/
│   ├── bindings.cpp        
│   └── softmax_kernel.cu   
├── my_ext/
│   └── __init__.py         
└── setup.py 
```

`bindings.cpp` 负责 pybind11 绑定，`softmax_kernel.cu` 放 CUDA kernel 和 C++ wrapper，Python 包只 import 编译出来的 `_C` 扩展。

### 最小 CMakeLists.txt

下面这个模板适合本地先跑通 3090（sm_86）上的 CUDA 扩展：

```cmake
cmake_minimum_required(VERSION 3.24)
project(my_ext LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")

find_package(Torch REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_C
    csrc/bindings.cpp
    csrc/softmax_kernel.cu
)

target_include_directories(_C PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES})
target_compile_features(_C PRIVATE cxx_std_17)
target_compile_definitions(_C PRIVATE TORCH_EXTENSION_NAME=_C)

set_target_properties(_C PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
)
```

`CMAKE_CUDA_ARCHITECTURES 86` 对应 RTX 3090。换显卡时要改成对应架构，比如 A100 是 80，H100 是 90。只写本机实验可以用固定值；要发给别人用时再考虑多架构编译。

### 构建命令

```bash
cmake -S . -B build
cmake --build build -j
```

如果是 Python 包，可以在 `setup.py` 或 `pyproject.toml` 里调用 CMake。早期阶段先手动 CMake 跑通，比一上来塞进打包流程更容易定位问题。

