---
order: 3
---

# CMake 构建实践

> 这是我学习用 CMake 管理 CUDA 项目过程中的笔记，记录实际操作和踩坑经历。

## 为什么要学 CMake

单文件用 `nvcc -o softmax softmax.cu` 就行，但真实项目（vLLM、FlashAttention）全部用 CMake 管理几十上百个源文件。看不懂 CMakeLists.txt 就没法在本地编译和修改这些项目。

## python 调用 cuda 算子
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


## 踩坑记录

1. find_package 找不到 PyTorch
设置 `CMAKE_PREFIX_PATH`
```cmake
execute_process(
    COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")
find_package(Torch REQUIRED)
```

2. pybind11 3.x + CMake 4.x 需要先 find_package(Python) 才能用 python_add_library
3. torch_python 库不在默认搜索路径，需要用完整路径链接
4. TORCH_EXTENSION_NAME 宏只在 torch.utils.cpp_extension 构建时才定义，用 CMake 构建时要手动写模块名 _C


## 还没搞懂的

- Generator Expression 语法看着很绕，需要更多实践
- CUDA_SEPARABLE_COMPILATION 什么时候真正需要开启
