项目结构
text
├── config/                    # 配置文件
│   ├── config.py              # 基础配置
│   ├── balanced_training_config.py    # 平衡训练配置
│   └── pure_diffusion_config.py       # 纯扩散模型配置
├── models/                    # 模型定义
│   ├── gnn_model.py           # 基础GNN模型
│   ├── improved_gnn_model.py  # 改进版GNN模型
│   └── ddpm_diffusion_model.py # 扩散模型
├── data_loader.py             # 数据加载与处理
├── main.py                    # 主训练管道
├── improved_pipeline.py       # 改进版训练管道
├── robust_pipeline.py         # 鲁棒性训练管道
├── enhanced_training_strategies.py  # 增强训练策略
└── evaluation/                # 评估模块
    ├── advanced_model_evaluation.py
    └── retest_models.py
工作流程
1. 数据准备阶段
使用ESM-2模型提取蛋白质序列的嵌入表示

将每个蛋白质转换为图结构（节点=残基，边=KNN连接）

标注结合位点信息（正样本）和非结合位点（负样本）

2. 扩散模型训练
仅使用训练集中的正样本训练扩散模型

学习正样本的分布特征

成为"正样本生成器"

3. 数据增强
对每个蛋白质图，计算需要生成的正样本数量

使用训练好的扩散模型生成新样本

进行质量控制和多样性筛选

将新样本合并到原始图中，重建图结构

4. GNN模型训练
使用增强后的数据训练图神经网络

支持多种GNN架构（GAT、GCN、GraphSAGE）

包含正则化、早停等机制

5. 评估与测试
在多测试集上进行全面评估

提供多种评估指标（F1、AUC-PR、MCC等）

生成详细的性能报告

快速开始
环境配置
bash
pip install torch torch-geometric esm
数据准备
将蛋白质数据文件放在Raw_data/目录下，格式为：

text
>protein_name
AminoAcidSequence
0001000100001000... (结合位点标签)
运行基础管道
bash
python main.py
运行改进版管道
bash
python improved_pipeline.py
运行鲁棒性管道
bash
python robust_pipeline.py
配置选项
项目提供多种配置策略：

基础配置 (config.py): 标准1:1平衡训练

平衡训练 (balanced_training_config.py): 针对测试性能优化

纯扩散配置 (pure_diffusion_config.py): 仅使用扩散模型增强

关键配置参数：

target_ratio: 目标正样本比例

diffusion_epochs: 扩散模型训练轮数

use_focal_loss: 是否使用Focal Loss

quality_threshold: 生成样本质量阈值

预期结果
运行完成后，项目将生成：

训练好的模型文件

增强后的数据集

详细的性能评估报告

不同配置的对比分析

技术亮点
扩散模型应用: 将先进的扩散模型技术应用于生物信息学领域

质量控制: 对生成样本进行严格的质量筛选

域适应: 提高模型在不同数据集间的泛化能力

集成学习: 结合多个模型提升最终性能

全面评估: 提供多角度的模型性能分析

注意事项
需要足够的GPU内存（建议16GB以上）

ESM模型加载需要一定时间

大规模蛋白质可能需要调整图大小限制

不同数据集可能需要调整目标比例参数
