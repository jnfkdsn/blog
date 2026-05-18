---
order: 1
title: Python 前置知识
updated: 2026-05-18
tags: [python, triton]
status: seed
---

# Python 前置知识

相关路线：[Triton 学习笔记](/notes/triton/) / [GPU 编程与算子优化知识地图](/notes/gpu-programming)

## dataclass
类装饰器，自动为数据类生成以下标准方法：
__init__：构造函数（初始化属性）
__repr__：友好的打印字符串（方便调试）
__eq__：对象相等性比较（按字段值对比）
它只用于存储数据，不适合写复杂业务逻辑。
