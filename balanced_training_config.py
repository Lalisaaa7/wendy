import torch


class BalancedTrainingConfig:
    """平衡训练配置 - 针对测试性能优化"""
    def __init__(self):
        # 基础配置
        self.seed = 42
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '6'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device} (Physical GPU 6)")

        # 数据配置
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(current_dir, "Raw_data")
        self.output_dir = os.path.join(current_dir, "Augmented_data_balanced")
        self.test_ratio = 0.2

        # 扩散模型配置
        self.diffusion_input_dim = 1280
        self.diffusion_T = 200
        self.diffusion_epochs = 60
        self.diffusion_batch_size = 64
        self.diffusion_lr = 1e-4

        # **关键改进**: 大幅降低目标平衡比例
        self.target_ratio = 0.15  # 从50%降到15% (接近真实测试分布)
        self.min_samples_per_protein = 10  # 减少最小生成样本
        self.knn_k = 3
        self.max_nodes_per_graph = 2000
        
        # 不使用过采样
        self.use_oversampling = False
        
        # GNN模型配置 - 增强正则化
        self.gnn_hidden_dim = 256  # 降低隐藏层维度减少过拟合
        self.gnn_epochs = 30  # 减少训练轮数
        self.gnn_lr = 5e-4  # 降低学习率
        self.gnn_dropout = 0.5  # 大幅增加dropout
        self.gnn_patience = 5  # 更早停止
        
        # 新增: 损失函数配置
        self.use_focal_loss = True
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
        self.pos_weight = 1.5  # 降低正样本权重 (从3.0到1.5)

        # 保存选项
        self.save_diffusion_model = True
        self.save_augmented_data = True

        print(f"🎯 平衡训练配置:")
        print(f"  - 目标比例: {self.target_ratio:.1%} (大幅降低)")
        print(f"  - 增强正则化: dropout={self.gnn_dropout}")
        print(f"  - 使用Focal Loss: {self.use_focal_loss}")
        print(f"  - 正样本权重: {self.pos_weight}")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")