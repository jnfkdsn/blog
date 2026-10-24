---
order: 40
title: 量化部署与评测
updated: 2026-10-17
---

# 量化部署与评测

本章整理自 [A Survey of Low-bit Large Language Models](https://doi.org/10.48550/arXiv.2409.16694) 的对应主题。全篇阅读顺序见[量化笔记](./)。

总结量化工具和评测。
LLMC、LMQuant、MI-optimize 等工具比较适合做多算法复现和公平比较；如果目标是跨推理框架部署，LLMC 更突出，因为它有灵活量化设置和多后端兼容。

评测指标分两大类：

### 效率

* deployability：能不能真正部署；
* throughput：吞吐；
* storage saving：存储节省；
* calibration time：PTQ 生产量化模型的时间成本。

理论上参数变小会加速，但实际是否加速取决于系统实现，所以 benchmark 必须测真实 throughput，而不能只看 bitwidth。

### 生成质量

包括：

* perplexity；
* accuracy；
* reasoning / logic；
* completion；
* trustworthiness；
* robustness；
* dialogue；
* long-context；
* multitask；
* safety。

---
