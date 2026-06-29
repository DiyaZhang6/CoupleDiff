import torch
import yaml
from pathlib import Path

# --- PyG Imports for creating dummy data ---
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch

# --- fvcore Imports for FLOPs calculation ---
from fvcore.nn import FlopCountAnalysis, flop_count_table

# --- 1. 导入您的模型和配置加载器 ---
from models.model import DynaModel


def load_config(config_path: Path) -> dict:
    """从指定的路径加载YAML配置文件。"""
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dummy_input_batch(config: dict, device: str = 'cpu') -> Batch:
    """
    创建一个与您的模型输入格式完全匹配的虚拟输入Batch对象。
    这是计算FLOPs最关键的部分。
    """
    print("--- 正在创建虚拟输入数据 ---")

    # --- 估计一个典型复合物的大小 (你需要根据你的数据集调整这些值) ---
    # 假设一个中等大小的蛋白质和配体
    num_backbone_nodes = 300  # 例如, 300个残基
    num_sidechain_atoms = 1500  # 例如, 1500个侧链原子
    num_drug_atoms = 30  # 例如, 30个药物重原子

    total_atoms = num_backbone_nodes + num_sidechain_atoms + num_drug_atoms
    print(
        f"虚拟样本尺寸: Backbone Nodes={num_backbone_nodes}, Sidechain Atoms={num_sidechain_atoms}, Drug Atoms={num_drug_atoms}")

    # 从config中获取特征维度
    backbone_in_dim = len(config['backbone_graph_task']['allowed_amino_acids']) + 2
    sidechain_in_dim = sum([
        len(config['sidechain_graph_task']['atom_symbols']), 8,
        len(config['sidechain_graph_task']['hybridization_types'])
    ])
    drug_in_dim = sum([
        len(config['drug_graph_task']['atom_symbols']), 1,
        len(config['drug_graph_task']['hybridization_types']),
        len(config['drug_graph_task']['chiral_tags']), 5
    ])
    embed_dim = config['interaction_params']['embed_dim']

    # 创建各个组件的Data对象
    backbone_data = Data(
        node_s=torch.randn(num_backbone_nodes, backbone_in_dim),
        node_v={'ca_coord': torch.randn(num_backbone_nodes, 3)},
        edge_index=torch.randint(0, num_backbone_nodes, (2, num_backbone_nodes * 8))  # 假设每个残基有8个邻居
    )
    sidechain_data = Data(
        node_s=torch.randn(num_sidechain_atoms, sidechain_in_dim),
        node_v=torch.randn(num_sidechain_atoms, 3),  # 注意: 您的代码这里似乎有误, node_v应该是3D坐标
        edge_index=torch.randint(0, num_sidechain_atoms, (2, num_sidechain_atoms * 4))
    )
    drug_data = Data(
        node_s=torch.randn(num_drug_atoms, drug_in_dim),
        node_v=torch.randn(num_drug_atoms, 3),
        edge_index=torch.randint(0, num_drug_atoms, (2, num_drug_atoms * 2))
    )

    # 创建一个异构图(HeteroData)的字典形式
    data_dict = {
        'backbone': backbone_data,
        'sidechain': sidechain_data,
        'drug': drug_data,
        # 添加交互边的s_matrix (LJ势能矩阵)
        ('backbone', 'interacts_with', 'sidechain'): {'s_matrix': torch.randn(num_backbone_nodes, num_sidechain_atoms)},
        ('backbone', 'interacts_with', 'drug'): {'s_matrix': torch.randn(num_backbone_nodes, num_drug_atoms)},
        ('sidechain', 'interacts_with', 'drug'): {'s_matrix': torch.randn(num_sidechain_atoms, num_drug_atoms)},
        # 添加其他必要的属性
        'r_init': torch.randn(total_atoms, 3),
        'r_true': torch.randn(total_atoms, 3),
        'atom_group_ids': torch.cat([
            torch.full((num_backbone_nodes,), 0),
            torch.full((num_sidechain_atoms,), 1),
            torch.full((num_drug_atoms,), 2)
        ]),
        'affinity': torch.tensor([7.5])  # 任意亲和力值
    }

    # 使用Batch.from_data_list来创建一个批次大小为1的Batch对象
    # 这是确保所有ptr和batch索引被正确创建的最简单方法
    batch = Batch.from_data_list([data_dict])

    print("虚拟输入数据创建成功。\n")
    return batch.to(device)


# --- 主执行函数 ---
if __name__ == '__main__':
    PROJECT_ROOT = Path("/home/zdy/Project2/")
    CONFIG_PATH = PROJECT_ROOT / "config.yaml"

    # 确定设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"将在设备上进行计算: {device}")

    try:
        # 1. 加载配置
        print(f"正在从 {CONFIG_PATH} 加载配置...")
        config = load_config(CONFIG_PATH)
        print("配置加载成功。")

        # 2. 初始化DynaModel并移到设备上
        print("正在初始化 DynaModel...")
        model = DynaModel(config).to(device)
        model.eval()  # 设置为评估模式
        print("模型初始化成功。")

        # 3. 创建与模型输入格式匹配的虚拟输入数据
        dummy_batch = create_dummy_input_batch(config, device)
        # 将forward的参数包装在一个元组中
        model_inputs = (dummy_batch,)

        # 4. 使用fvcore进行FLOPs分析
        print("--- 开始进行FLOPs分析 ---")
        flop_analyzer = FlopCountAnalysis(model, model_inputs)

        # 打印详细的、按模块分的FLOPs表格
        print("\n详细FLOPs分析 (按模块):")
        print(flop_count_table(flop_analyzer))

        # 获取总数
        total_flops = flop_analyzer.total()
        total_flops_g = total_flops / 1e9  # 转换为GFLOPs

        print("\n--- FLOPs 总结 ---")
        print(f"模型总FLOPs: {total_flops:,}")
        print(f"模型总GFLOPs: {total_flops_g:.2f} GFLOPs")
        print("--------------------")

    except FileNotFoundError as e:
        print(f"\n错误: {e}")
    except KeyError as e:
        print(f"\n错误: 配置文件中缺少必需的键: {e}")
    except Exception as e:
        print(f"\n发生未知错误: {e}")