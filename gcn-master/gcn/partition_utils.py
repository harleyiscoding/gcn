import numpy as np
import scipy.sparse as sp
import pymetis
from functools import lru_cache

# --- METIS静态划分 ---
def adj_to_metis(adj):
    """将scipy稀疏邻接矩阵转为pymetis邻接表格式"""
    adj = adj.tolil()
    n = adj.shape[0]
    neighbors = []
    for i in range(n):
        neighbors.append(list(adj.rows[i]))
    return neighbors

def metis_partition(adj, k):
    """使用pymetis对邻接矩阵做k-way划分，返回每个节点的分区号数组"""
    neighbors = adj_to_metis(adj)
    _, parts = pymetis.part_graph(k, adjacency=neighbors)
    return np.array(parts)

def get_partition_masks(part_labels, num_partitions):
    """根据分区标签生成每个分区的节点索引列表"""
    return [np.where(part_labels == i)[0] for i in range(num_partitions)]

# --- 分区感知特征归一化 ---
def preprocess_features_by_partition(features, partition_masks):
    """按分区进行特征归一化，返回分区拼接后的特征矩阵"""
    partitioned_features = []
    for mask in partition_masks:
        part_feat = features[mask]
        rowsum = np.array(part_feat.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        partitioned_features.append(r_mat_inv.dot(part_feat))
    return sp.vstack(partitioned_features)

# --- 跨分区通信边界节点识别 ---
class PartitionCommunicator:
    def __init__(self, adj, partition_masks):
        self.boundary_nodes = self._find_boundary_nodes(adj, partition_masks)
    def _find_boundary_nodes(self, adj, masks):
        boundary_sets = []
        for i, mask_i in enumerate(masks):
            neighbors = set()
            for j, mask_j in enumerate(masks):
                if i != j:
                    sub_adj = adj[mask_i][:, mask_j]
                    neighbors.update(mask_i[np.unique(sub_adj.nonzero()[0])])
            boundary_sets.append(neighbors)
        return boundary_sets

# --- 分区邻接矩阵缓存 ---
@lru_cache(maxsize=8)
def get_partition_adj(adj, part_id, partition_masks_tuple):
    # partition_masks_tuple: tuple of tuples for lru_cache hashability
    mask = np.array(partition_masks_tuple[part_id])
    return adj[mask][:, mask]

# --- PIM Bank映射 ---
def map_to_pim_banks(features, adj, partition_masks, num_banks):
    """将分区映射到PIM Bank，返回每个bank的节点索引和邻接矩阵切片"""
    bank_mappings = []
    # lru_cache要求tuple类型
    partition_masks_tuple = tuple([tuple(m) for m in partition_masks])
    for part_id in range(len(partition_masks)):
        bank_id = part_id % num_banks
        bank_mappings.append({
            'bank_id': bank_id,
            'node_indices': partition_masks[part_id],
            'adj_slice': get_partition_adj(adj, part_id, partition_masks_tuple)
        })
    return bank_mappings

def extract_subgraph_data(adj, features, y_train, y_val, y_test,
                        train_mask, val_mask, test_mask, part_nodes):
    """提取单个分区的子图数据，返回dict。"""
    adj_sub = adj[part_nodes, :][:, part_nodes]
    features_sub = features[part_nodes, :]
    y_train_sub = y_train[part_nodes]
    y_val_sub = y_val[part_nodes]
    y_test_sub = y_test[part_nodes]
    train_mask_sub = train_mask[part_nodes]
    val_mask_sub = val_mask[part_nodes]
    test_mask_sub = test_mask[part_nodes]
    return {
        'adj_sub': adj_sub,
        'features_sub': features_sub,
        'y_train_sub': y_train_sub,
        'y_val_sub': y_val_sub,
        'y_test_sub': y_test_sub,
        'train_mask_sub': train_mask_sub,
        'val_mask_sub': val_mask_sub,
        'test_mask_sub': test_mask_sub
    }

def extract_all_partition_subgraphs(adj, features, y_train, y_val, y_test,
                                    train_mask, val_mask, test_mask, partition_masks):
    """提取所有分区的子图数据，返回列表。"""
    subgraph_list = []
    for part_nodes in partition_masks:
        subgraph_data = extract_subgraph_data(
            adj, features, y_train, y_val, y_test,
            train_mask, val_mask, test_mask, part_nodes)
        subgraph_list.append(subgraph_data)
    return subgraph_list

# --- CSR+Delta+Varint压缩工具 ---
def delta_encode(arr):
    """对一维int数组做delta编码"""
    if len(arr) == 0:
        return np.array([], dtype=np.int32)
    return np.array([arr[0]] + [arr[i] - arr[i-1] for i in range(1, len(arr))], dtype=np.int32)

def delta_decode(arr):
    """对delta编码还原"""
    if len(arr) == 0:
        return np.array([], dtype=np.int32)
    out = [arr[0]]
    for d in arr[1:]:
        out.append(out[-1] + d)
    return np.array(out, dtype=np.int32)

def varint_encode_number(n):
    """对单个正整数做Varint编码，返回字节列表"""
    out = []
    while True:
        to_write = n & 0x7F
        n >>= 7
        if n:
            out.append(to_write | 0x80)
        else:
            out.append(to_write)
            break
    return out

def varint_encode(arr):
    """对一组正整数做Varint编码，返回字节列表"""
    out = []
    for n in arr:
        out.extend(varint_encode_number(int(n)))
    return bytes(out)

def varint_decode(data):
    """Varint解码，返回整数列表"""
    out = []
    n = 0
    shift = 0
    for b in data:
        n |= (b & 0x7F) << shift
        if b & 0x80:
            shift += 7
        else:
            out.append(n)
            n = 0
            shift = 0
    return out

def compress_csr_with_delta_varint(sparse_mat):
    """
    对scipy.sparse.csr_matrix的indices做行内delta+varint压缩
    返回: indptr, indices_varint_bytes, data
    用法：
        indptr, indices_bytes, data = compress_csr_with_delta_varint(adj_sub)
    """
    assert sp.isspmatrix_csr(sparse_mat)
    indptr = sparse_mat.indptr
    indices = sparse_mat.indices
    data = sparse_mat.data
    indices_varint_bytes = bytearray()
    for i in range(len(indptr) - 1):
        row_indices = indices[indptr[i]:indptr[i+1]]
        delta = delta_encode(row_indices)
        indices_varint_bytes.extend(varint_encode(delta))
    return indptr, bytes(indices_varint_bytes), data

def decompress_csr_with_delta_varint(indptr, indices_varint_bytes, data, shape):
    """
    解压缩为scipy.sparse.csr_matrix
    用法：
        adj_sub_restored = decompress_csr_with_delta_varint(indptr, indices_bytes, data, adj_sub.shape)
    """
    indices = []
    ptr = 0
    for i in range(len(indptr) - 1):
        row_len = indptr[i+1] - indptr[i]
        row_deltas = []
        cnt = 0
        while cnt < row_len:
            n = 0
            shift = 0
            while True:
                b = indices_varint_bytes[ptr]
                ptr += 1
                n |= (b & 0x7F) << shift
                if not (b & 0x80):
                    break
                shift += 7
            row_deltas.append(n)
            cnt += 1
        row_indices = delta_decode(row_deltas)
        indices.extend(row_indices)
    indices = np.array(indices, dtype=np.int32)
    return sp.csr_matrix((data, indices, indptr), shape=shape)

# ---
# 用法说明：
# 对训练时的adj_sub、features_sub等稀疏矩阵（scipy.sparse.csr_matrix）
#   indptr, indices_bytes, data = compress_csr_with_delta_varint(adj_sub)
#   adj_sub_restored = decompress_csr_with_delta_varint(indptr, indices_bytes, data, adj_sub.shape)
# features_sub同理。 

def auto_compress_features(features):
    """
    自动判断特征类型并选择最佳压缩方式：
    - 稀疏矩阵：CSR+Delta+Varint
    - 稠密且全为0/1：转CSR+Delta+Varint
    - 稠密且非0/1：量化为uint8，返回min/max
    返回dict，包含压缩类型和必要参数
    """
    if sp.issparse(features):
        # 稀疏特征直接用CSR+Delta+Varint
        features_csr = features.tocsr()
        indptr, indices_bytes, data = compress_csr_with_delta_varint(features_csr)
        return {
            'type': 'csr_delta_varint',
            'indptr': indptr,
            'indices_bytes': indices_bytes,
            'data': data,
            'shape': features.shape
        }
    elif np.issubdtype(features.dtype, np.integer) and np.all((features == 0) | (features == 1)):
        # 稠密0/1特征，转CSR
        features_csr = sp.csr_matrix(features)
        indptr, indices_bytes, data = compress_csr_with_delta_varint(features_csr)
        return {
            'type': 'csr_delta_varint',
            'indptr': indptr,
            'indices_bytes': indices_bytes,
            'data': data,
            'shape': features.shape
        }
    else:
        # 稠密且非0/1，量化为uint8
        f_min, f_max = features.min(), features.max()
        quantized = np.round((features - f_min) / (f_max - f_min) * 255).astype(np.uint8)
        return {
            'type': 'quant_uint8',
            'quantized': quantized,
            'min': f_min,
            'max': f_max,
            'shape': features.shape
        }

def auto_decompress_features(compressed):
    """
    自动解压/反量化auto_compress_features的输出
    """
    if compressed['type'] == 'csr_delta_varint':
        return decompress_csr_with_delta_varint(
            compressed['indptr'], compressed['indices_bytes'], compressed['data'], compressed['shape'])
    elif compressed['type'] == 'quant_uint8':
        q = compressed['quantized'].astype(np.float32)
        f_min, f_max = compressed['min'], compressed['max']
        return (q / 255.0) * (f_max - f_min) + f_min
    else:
        raise ValueError('Unknown compressed feature type: ' + str(compressed['type'])) 