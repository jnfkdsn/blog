# AI 系统学习记录

基于 VitePress 的个人知识库，内容围绕编程基础、编译器、算子优化和模型推理组织。

## 内容入口

- [知识库首页](docs/notes/index.md)与[学习路线](docs/notes/roadmap.md)。
- [C++ 工程基础](docs/notes/cpp/index.md)。
- [编译器](docs/notes/compile/index.md)：传统基础、AI 编译基础、MLIR、TVM、PyTorch Compiler、Triton、NPU IR。
- [算子编程与优化](docs/notes/kernels/index.md)：CUDA、Ascend C、算子算法。
- [模型与推理系统](docs/notes/inference/index.md)：模型计算、推理引擎与量化。

## 归档规则

目录表示知识归属，模块首页维护阅读顺序。跨领域主题保留一个主要解释位置，通过链接连接其他路线；Triton 的编程与编译机制统一放在 `compile/triton/`。

`notes/` 保存可独立阅读的原理，`posts/` 保存具体实现和性能复盘，`projects/` 保存项目设计与阶段成果。正文需要的代码和关键结果直接展示，不要求读者打开配套实验才能理解。

总路线说明领域关系，课程首页说明篇章顺序，覆盖表说明范围与证据。个人阅读进度保存在父工作区的 `学习进度.md`，不进入公开课程的当前阶段描述。

只有实际内容形成模块后才创建子目录；移动页面时同步站内链接、图片、实验仓库引用和文档审计脚本，不保留旧地址占位页。旧外部书签不会自动跳转。

## 本地验证

在本仓库根目录执行：

```bash
npm run build
python3 scripts/check_site.py
```

构建检查 Markdown 与站内页面目标；第二条命令检查生成页面的链接、锚点和静态资源，包含 GitHub Pages 的 `/blog/` 前缀，不请求外部网站。站点通过不代表课程中的 GPU/NPU 示例已执行。

## 配套实验

实验在独立的 [aicompiler 仓库](https://github.com/jnfkdsn/aicompiler)维护。站内链接使用 Markdown 路径，跨仓库源码链接使用 GitHub 地址；发布前需确保两边的新文件都已上传。
