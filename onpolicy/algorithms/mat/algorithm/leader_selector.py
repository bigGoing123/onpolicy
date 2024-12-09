import logging
import sklearn.cluster
import numpy as np
import torch


def assign_leaders(positions, n_clusters, seq_len):
    """
    使用 K-means 聚类选择领导者并划分区域
    :param positions: 智能体位置 (N, 2) 的数组
    :param n_clusters: 区域数量（或领导者数量）
    :param seq_len: 序列长度（即智能体数量）
    :return: leaders 的索引和每个智能体所属区域的标签
    """
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=0).fit(positions)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    leaders = []

    # 为每个聚类选择一个领导者
    for i in range(n_clusters):
        cluster_points = np.where(labels == i)[0]
        distances = np.linalg.norm(positions[cluster_points] - centers[i], axis=1)
        leader = cluster_points[np.argmin(distances)]
        # 确保领导者索引不超过 seq_len
        if leader >= seq_len:
            leader = seq_len - 1  # 如果索引超出范围，强制将其设置为最大有效索引
        leaders.append(leader)

    logging.info(f"Clustering results:\\nCenters: {centers}\\nLabels: {labels}\\nLeaders: {leaders}")
    return leaders, labels


def generate_local_comm_mask(batch_size, num_heads, seq_len, leaders):
    """
    根据领导者-跟随者框架生成局部通信掩码
    :param batch_size: 批量大小
    :param num_heads: 注意力头数
    :param seq_len: 序列长度（即智能体数量）
    :param leaders: 领导者的索引
    :return: 局部通信掩码
    """
    # 初始化通信掩码，大小为 [batch_size, num_heads, seq_len, seq_len]
    mask_condition = np.zeros((batch_size, num_heads, seq_len, seq_len), dtype=bool)

    # 对每个领导者和跟随者进行局部通信
    for leader in leaders:
        mask_condition[:, :, leader, :] = True  # 领导者可以和所有人通信
        mask_condition[:, :, :, leader] = True  # 所有人都可以和领导者通信

    return torch.tensor(mask_condition, dtype=torch.bool)


def apply_local_communication(att, positions, n_clusters, batch_size, num_heads, seq_len):
    """
    应用局部通信机制，使用领导者-跟随者框架
    :param att: 原始注意力矩阵 [batch_size, num_heads, seq_len, seq_len]
    :param positions: 智能体的位置 (batch_size, num_agents, 2, num_timesteps)
    :param n_clusters: 领导者数量
    :param batch_size: 批量大小
    :param num_heads: 注意力头数
    :param seq_len: 序列长度（即智能体数量）
    :return: 局部通信后的注意力矩阵
    """
    # Debugging: print positions to track if it's None
    # print(
    #     f"Debug: positions shape before apply_local_communication: {positions.shape if positions is not None else 'None'}")

    if positions is None:
        print("Error: positions is None in apply_local_communication!")
        raise ValueError("positions should not be None!")

    # 更新


    # Reshape positions to a 2D array (batch_size * num_agents, num_timesteps)
    positions_np = positions.view(-1, 2).cpu().numpy()

    # 获取领导者和标签
    leaders, _ = assign_leaders(positions_np, n_clusters, seq_len)

    # 根据领导者选择生成局部通信掩码
    mask_condition = generate_local_comm_mask(batch_size, num_heads, seq_len, leaders)

    # Debugging: print out shapes to verify
    # print(f"att shape: {att.shape}")
    # print(f"mask_condition shape: {mask_condition.shape}")

    # Ensure mask_condition has the same shape as att
    if mask_condition.shape != att.shape:
        # print(f"Reshaping mask_condition from {mask_condition.shape} to {att.shape}")
        mask_condition = mask_condition.view(batch_size, num_heads, seq_len, seq_len)

    # Ensure that mask_condition is a boolean tensor for correct masking
    mask_condition = mask_condition.to(dtype=torch.bool)

    # 创建一个新的矩阵来保存修改后的注意力矩阵
    masked_att = att.clone()  # Clone to avoid modifying the original att

    # Apply the mask: mask_condition should be True for allowed communication and False for blocked communication
    # 如果mask_condition为True，则保留原值，否则将值设为负无穷
    for b in range(batch_size):
        for h in range(num_heads):
            # Apply the mask to each head and batch element
            masked_att[b, h, ~mask_condition[b, h]] = float('-inf')

    return masked_att
