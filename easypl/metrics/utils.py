import torch

available_distances = ['L2', 'Normalize cosine', 'Cosine']


def build_distance(distance_name: str):
    if distance_name not in available_distances:
        raise ValueError(f'Distance {distance_name} is not available. Choice from {available_distances}')
    if distance_name == 'L2':
        return l2_pairwaise
    elif distance_name == 'Normalize cosine':
        return cosine_pairwaise_normalize
    elif distance_name == 'Cosine':
        return cosine_pairwaise


def l2_pairwaise(query: torch.Tensor, data: torch.Tensor):
    return torch.cdist(query, data, p=2)


def cosine_pairwaise_normalize(query: torch.Tensor, data: torch.Tensor):
    return torch.matmul(query, data.t())


def cosine_pairwaise(query: torch.Tensor, data: torch.Tensor):
    mat = cosine_pairwaise_normalize(query, data)
    return mat / query.norm(dim=1).unsqueeze(-1).repeat(1, mat.size(1)) / data.norm(dim=1).unsqueeze(0).repeat(
        mat.size(0), 1)