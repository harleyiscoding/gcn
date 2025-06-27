import pickle
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt
from partition_utils import metis_partition, get_partition_masks, extract_all_partition_subgraphs
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_data

# 参数
DATASET = 'cora'  # 可改为citeseer/pubmed
data_dir = 'data'
num_parts = 4  # 分区数


def visualize_partition(graph, part_labels, out_path):
    G = nx.Graph()
    for node, neighbors in graph.items():
        for n in neighbors:
            G.add_edge(node, n)
    # 颜色映射
    color_map = [part_labels[node] if node < len(part_labels) else 0 for node in G.nodes()]
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color=color_map, cmap=plt.cm.tab10, node_size=30)
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)
    plt.title(f"{DATASET} Partition Visualization (k={num_parts})")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Partition visualization saved to {out_path}")
    plt.close()

def main():
    # 1. 读取邻接表
    graph_path = f"{data_dir}/ind.{DATASET}.graph"
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f, encoding='latin1')
    # 1.5 通过load_data获取adj
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(DATASET)
    print(f"adj type: {type(adj)}")
    print("adj shape:", adj.shape)
    print("前5行非零元素（行, 列, 值）:")
    adj_coo = adj.tocoo()
    for i in range(5):
        row_idx = adj_coo.row[i]
        col_idx = adj_coo.col[i]
        val = adj_coo.data[i]
        print(f"({row_idx}, {col_idx}): {val}")
    # 2. 转为邻接矩阵
    adj2 = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # 3. METIS划分
    part_labels = metis_partition(adj2, num_parts)
    partition_masks = get_partition_masks(part_labels, num_parts)
    # 4. 打印分区统计
    print(f"=== {DATASET} 分区统计 (共{num_parts}分区) ===")
    for i, mask in enumerate(partition_masks):
        print(f"Partition {i}: {len(mask)} nodes, node indices: {mask[:10]}{'...' if len(mask)>10 else ''}")

    # === 集成：提取每个分区的子图数据 ===
    subgraph_list = extract_all_partition_subgraphs(
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, partition_masks)
    print(f"subgraph_list 长度: {len(subgraph_list)}")
    print(f"subgraph_list[0] keys: {list(subgraph_list[0].keys())}")
    # 5. 统计跨分区边数
    cross_edges = 0
    for i, mask in enumerate(partition_masks):
        for node in mask:
            neighbors = adj2.getrow(node).nonzero()[1]
            for n in neighbors:
                if part_labels[n] != i:
                    cross_edges += 1
    print(f"Total cross-partition edges: {cross_edges}")
    # 6. 可视化
    out_dir = 'data_metis'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'partition_viz_{DATASET}_k{num_parts}.png')
    visualize_partition(graph, part_labels, out_path)

if __name__ == '__main__':
    main() 