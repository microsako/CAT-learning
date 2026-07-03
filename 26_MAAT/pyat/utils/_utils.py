def make_hot_vector(indices, num_dim):
    """构造多热(multi-hot)向量

    Args:
        indices: list,值为 1 的位置下标
        num_dim: int,向量总长度

    Returns:
        v: list,多热向量
    """
    v = [0] * num_dim
    for i in indices:
        v[i] = 1
    return v
