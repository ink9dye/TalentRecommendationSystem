import argparse
import os

def parse_kgat_args():
    parser = argparse.ArgumentParser(description="Run KGAT-AX.")

    # 1. 基础实验配置
    parser.add_argument('--seed', type=int, default=2019,
                        help='Random seed.')
    # 必须与 generate_training_data.py 中的输出目录名保持一致
    parser.add_argument('--data_name', nargs='?', default='kgatax_train_data',
                        help='训练数据所在的子目录名称')
    # 指向包含训练数据的根目录
    parser.add_argument('--data_dir', nargs='?', default='data/',
                        help='Input data path.')

    # 2. 预训练配置 (初次运行建议设为 0)
    parser.add_argument('--use_pretrain', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with stored model.')
    parser.add_argument('--pretrain_embedding_dir', nargs='?', default='datasets/pretrain/',
                        help='Path of learned embeddings.')
    parser.add_argument('--pretrain_model_path', nargs='?', default='trained_model/model.pth',
                        help='Path of stored model.')

    # 3. 训练 Batch 大小 (针对 16GB 共享内存优化)
    # 在 CPU 训练模式下，较大的 Batch 可以减少迭代次数，但需注意内存压力
    parser.add_argument('--cf_batch_size', type=int, default=1024,
                        help='CF batch size.')
    parser.add_argument('--kg_batch_size', type=int, default=2048,
                        help='KG batch size.')
    parser.add_argument('--test_batch_size', type=int, default=256,
                        help='Test batch size.')

    # 4. 模型超参数
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='User / entity Embedding size.')
    parser.add_argument('--relation_dim', type=int, default=64,
                        help='Relation Embedding size.')
    parser.add_argument('--laplacian_type', type=str, default='random-walk',
                        help='Specify the type of the adjacency (laplacian) matrix.')
    parser.add_argument('--aggregation_type', type=str, default='bi-interaction',
                        help='Specify the type of the aggregation layer {gcn, graphsage, bi-interaction}.')
    # 三层传播：第一层聚合 1-hop，最后一层捕获 3-order connectivity
    parser.add_argument('--conv_dim_list', nargs='?', default='[64, 32, 16]',
                        help='Output sizes of every aggregation layer.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1, 0.1]',
                        help='Dropout probability.')

    # 5. 核心 AX (Academic Metrics) 插件参数
    # 学术指标包含：H-index, Citations, Works Count
    parser.add_argument('--n_aux_features', type=int, default=3,
                        help='学术指标特征维度 (H-index, Citations, Works).')
    # 修改：直接指向你实际存在的学术数据库 academic_dataset_v5.db
    parser.add_argument('--db_path', type=str,
                        default='E:/PythonProject/TalentRecommendationSystem/data/academic_dataset_v5.db',
                        help='SQLite 数据库的绝对路径，用于加载全息嵌入所需的辅助信息。')

    # 6. 损失函数与优化参数
    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')
    parser.add_argument('--cf_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating CF l2 loss.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--n_epoch', type=int, default=1000,
                        help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=10,
                        help='Number of epoch for early stopping.')

    # 7. 日志与评估配置
    parser.add_argument('--cf_print_every', type=int, default=1,
                        help='Iter interval of printing CF loss.')
    parser.add_argument('--kg_print_every', type=int, default=1,
                        help='Iter interval of printing KG loss.')
    # 考虑到 CPU 训练评估较慢，建议设为 5 或 10
    parser.add_argument('--evaluate_every', type=int, default=20,
                        help='Epoch interval of evaluating recall@K and ndcg@K.')
    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='Calculate metric@K when evaluating.')

    args = parser.parse_args()

    # 自动生成存储目录名，确保存储路径唯一
    save_dir = 'trained_model/KGAT/{}/embed-dim{}_relation-dim{}_{}_{}_{}_lr{}_pretrain{}/'.format(
        args.data_name, args.embed_dim, args.relation_dim, args.laplacian_type, args.aggregation_type,
        '-'.join([str(i) for i in eval(args.conv_dim_list)]), args.lr, args.use_pretrain)
    args.save_dir = save_dir

    return args