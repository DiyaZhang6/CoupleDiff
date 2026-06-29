#!/usr/bin/env python
# /home/zdy/Project2/models/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Dict, List


def _calculate_pi_pi_geometry(pred_coords: torch.Tensor, ring_pair_indices: List):
    if not ring_pair_indices or len(ring_pair_indices) == 0:
        return {
            'distances': torch.tensor([], device=pred_coords.device),
            'angles_rad': torch.tensor([], device=pred_coords.device),
            'displacements': torch.tensor([], device=pred_coords.device),
            'normals1': torch.tensor([], device=pred_coords.device),
            'normals2': torch.tensor([], device=pred_coords.device)
        }

    centroids1, centroids2 = [], []
    normals1, normals2 = [], []

    for r1_indices, r2_indices in ring_pair_indices:
        r1_coords = pred_coords[r1_indices]
        r2_coords = pred_coords[r2_indices]

        c1 = torch.mean(r1_coords, dim=0)
        c2 = torch.mean(r2_coords, dim=0)
        centroids1.append(c1)
        centroids2.append(c2)

        v1_1 = r1_coords[1] - r1_coords[0]
        v1_2 = r1_coords[-1] - r1_coords[0]
        n1 = F.normalize(torch.cross(v1_1, v1_2), p=2, dim=-1, eps=1e-8)

        v2_1 = r2_coords[1] - r2_coords[0]
        v2_2 = r2_coords[-1] - r2_coords[0]
        n2 = F.normalize(torch.cross(v2_1, v2_2), p=2, dim=-1, eps=1e-8)

        normals1.append(n1)
        normals2.append(n2)

    centroids1 = torch.stack(centroids1)
    centroids2 = torch.stack(centroids2)
    normals1 = torch.stack(normals1)
    normals2 = torch.stack(normals2)

    vec_c1_c2 = centroids2 - centroids1
    distances = torch.linalg.norm(vec_c1_c2, dim=-1)

    cos_theta = torch.abs(torch.sum(normals1 * normals2, dim=-1)).clamp(0.0, 1.0 - 1e-7)
    angles_rad = torch.acos(cos_theta)

    proj_on_n1 = torch.sum(vec_c1_c2 * normals1, dim=-1).unsqueeze(-1) * normals1
    displacements = torch.linalg.norm(vec_c1_c2 - proj_on_n1, dim=-1)

    return {
        'distances': distances, 'angles_rad': angles_rad,
        'displacements': displacements, 'normals1': normals1, 'normals2': normals2
    }


def loss_bond_length(pred_lengths, ref_lengths):
    if pred_lengths.numel() == 0: return torch.tensor(0.0, device=pred_lengths.device)
    loss = ((pred_lengths - ref_lengths) / (ref_lengths + 1e-6)) ** 2
    return torch.sum(loss)


def loss_bond_angle(pred_angles_rad, ref_angles_rad):
    if pred_angles_rad.numel() == 0: return torch.tensor(0.0, device=pred_angles_rad.device)
    return torch.sum((pred_angles_rad - ref_angles_rad) ** 2)


def loss_dihedral_angle(pred_angles_rad, true_angles_rad):
    if pred_angles_rad.numel() == 0: return torch.tensor(0.0, device=pred_angles_rad.device)
    return torch.sum((pred_angles_rad - true_angles_rad) ** 2)


def loss_vdw_repulsion(pred_coords, pair_indices, radii_values, params: Dict):
    """
    radii_values: 已经在 combine.py 中映射好的半径数据
    """
    if pair_indices is None or pair_indices.numel() == 0 or radii_values is None or radii_values.numel() == 0:
        return torch.tensor(0.0, device=pred_coords.device)
    output_device = pred_coords.device
    compute_device = params.get('compute_device')
    if compute_device == 'cpu':
        pred_coords = pred_coords.to('cpu')
        pair_indices = pair_indices.to('cpu')
        radii_values = radii_values.to('cpu')
    chunk_size = int(params.get('chunk_size', 8192))
    use_checkpoint = bool(params.get('use_checkpoint', True)) and pred_coords.device.type != 'cpu'
    total = pred_coords.new_tensor(0.0)
    for start in range(0, pair_indices.size(0), chunk_size):
        idx = pair_indices[start:start + chunk_size]
        def chunk_loss(coords):
            v = coords[idx[:, 1]] - coords[idx[:, 0]]
            distances = torch.sqrt(torch.sum(v ** 2, dim=-1) + 1e-8)
            radii_sum = radii_values[idx[:, 0]] + radii_values[idx[:, 1]]
            violations = F.relu(radii_sum - distances)
            return torch.sum(violations ** 2)
        if use_checkpoint and pred_coords.requires_grad:
            total = total + checkpoint(chunk_loss, pred_coords)
        else:
            total = total + chunk_loss(pred_coords)
    return total.to(output_device)

    # 注意：如果 radii_values 是每个原子的半径，则需按索引求和；
    # 如果 combine.py 已经存了 radii_sum，则直接使用。
    # 这里假设 radii_values 是原始 vdw_radii 数组


def loss_electrostatic_dipole(pred_coords, dipole_vectors, pair_indices, params: Dict):
    if pair_indices is None or pair_indices.numel() == 0 or dipole_vectors is None or dipole_vectors.numel() == 0:
        return torch.tensor(0.0, device=pred_coords.device)
    output_device = pred_coords.device
    compute_device = params.get('compute_device')
    if compute_device == 'cpu':
        pred_coords = pred_coords.to('cpu')
        pair_indices = pair_indices.to('cpu')
        dipole_vectors = dipole_vectors.to('cpu')
    dipole_vectors = torch.nan_to_num(dipole_vectors.float(), nan=0.0, posinf=0.0, neginf=0.0)
    chunk_size = int(params.get('chunk_size', 8192))
    use_checkpoint = bool(params.get('use_checkpoint', True)) and pred_coords.device.type != 'cpu'
    eps = params.get('distance_epsilon', 1e-6)
    total = pred_coords.new_tensor(0.0)
    for start in range(0, pair_indices.size(0), chunk_size):
        idx = pair_indices[start:start + chunk_size]
        p_i = dipole_vectors[idx[:, 0]]
        p_j = dipole_vectors[idx[:, 1]]
        def chunk_loss(coords):
            r_i, r_j = coords[idx[:, 0]], coords[idx[:, 1]]
            r_ij = r_i - r_j
            dist = torch.linalg.norm(r_ij, dim=-1).clamp_min(eps)
            r_hat = r_ij / dist.unsqueeze(-1)
            numerator = (
                torch.sum(p_i * p_j, dim=-1)
                - 3.0 * torch.sum(p_i * r_hat, dim=-1) * torch.sum(p_j * r_hat, dim=-1)
            )
            energy = numerator / (dist ** 3)
            return torch.sum(energy ** 2)
        if use_checkpoint and pred_coords.requires_grad:
            total = total + checkpoint(chunk_loss, pred_coords)
        else:
            total = total + chunk_loss(pred_coords)
    return total.to(output_device)


def loss_hydrogen_bond(pred_coords, hbond_triplets, params: Dict):
    if hbond_triplets is None or hbond_triplets.numel() == 0:
        return torch.tensor(0.0, device=pred_coords.device)
    d, h, a = pred_coords[hbond_triplets[:, 0]], pred_coords[hbond_triplets[:, 1]], pred_coords[hbond_triplets[:, 2]]
    da_dist = torch.linalg.norm(d - a, dim=-1)
    loss_dist = F.relu(da_dist - params['dist_max']) ** 2 + F.relu(params['dist_min'] - da_dist) ** 2
    vec_hd_n = F.normalize(d - h, p=2, dim=-1, eps=1e-8)
    vec_ha_n = F.normalize(a - h, p=2, dim=-1, eps=1e-8)
    theta_rad = torch.acos(torch.sum(vec_hd_n * vec_ha_n, dim=-1).clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    theta_deg = torch.rad2deg(theta_rad)
    loss_angle = F.relu(params['angle_min_deg'] - theta_deg) ** 2 + F.relu(theta_deg - params['angle_max_deg']) ** 2
    return torch.sum(loss_dist + loss_angle)


def loss_pi_pi_stacking(pred_geom: Dict, params: Dict):
    if pred_geom['distances'].numel() == 0: return torch.tensor(0.0, device=pred_geom['distances'].device)
    loss_dist = F.relu(pred_geom['distances'] - params['dist_max']) ** 2 + F.relu(
        params['dist_min'] - pred_geom['distances']) ** 2
    loss_angle = F.relu(torch.rad2deg(pred_geom['angles_rad']) - params['angle_max_deg']) ** 2
    loss_disp = F.relu(pred_geom['displacements'] - params['disp_max']) ** 2
    cos_theta = torch.sum(pred_geom['normals1'] * pred_geom['normals2'], dim=-1)
    alignment = (1 - torch.abs(cos_theta)) ** 2
    return torch.sum(loss_dist + loss_angle + loss_disp + alignment)


class TotalLoss(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config['total_loss']
        self.params = self.config['params']
        self.static_weights = self.config.get(
            'static_weights',
            {'L_noise': 0.0, 'L_structure': 1.0, 'L_phy_geo': 1.0},
        )

    def _clean_component(self, name: str, value: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(value, nan=0.0, posinf=1.0e6, neginf=1.0e6)

    def forward(self, outputs: Dict, batch):
        pred_coords = outputs.get('pred_coords')
        true_coords = outputs.get('true_coords')
        pred_noise = outputs.get('pred_noise')
        true_noise = outputs.get('true_noise')

        if pred_coords is None: return {
            'total_loss': torch.tensor(0.0, requires_grad=True, device=batch['backbone'].x.device)}
        device = pred_coords.device
        pred_coords = pred_coords.float()
        true_coords = true_coords.float() if true_coords is not None else None
        pred_noise = pred_noise.float() if pred_noise is not None else None
        true_noise = true_noise.float() if true_noise is not None else None

        w_noise = self.static_weights.get('L_noise', 0.0)
        w_struct = self.static_weights.get('L_structure', 1.0)
        w_phy = self.static_weights.get('L_phy_geo', 1.0)

        L_noise = F.mse_loss(torch.nan_to_num(pred_noise), true_noise) if (
                    w_noise > 0 and pred_noise is not None) else torch.tensor(0.0, device=device)
        L_structure = F.mse_loss(torch.nan_to_num(pred_coords),
                                 true_coords) if true_coords is not None else torch.tensor(0.0, device=device)

        phy = getattr(batch, 'phy_geo_labels', {})

        # Bond
        b_data = phy.get('bond', {})
        if b_data and b_data.get('indices') is not None and b_data['indices'].numel() > 0:
            idx = b_data['indices']
            dist = torch.sqrt(torch.sum((pred_coords[idx[:, 1]] - pred_coords[idx[:, 0]]) ** 2, dim=-1) + 1e-8)
            L_bond = loss_bond_length(dist, b_data['values'])  # 对齐 combine.py 的 values
        else:
            L_bond = torch.tensor(0.0, device=device)

        # Angle
        a_data = phy.get('angle', {})
        if a_data and a_data.get('indices') is not None and a_data['indices'].numel() > 0:
            idx = a_data['indices']
            v1, v2 = pred_coords[idx[:, 0]] - pred_coords[idx[:, 1]], pred_coords[idx[:, 2]] - pred_coords[idx[:, 1]]
            cos = torch.sum(F.normalize(v1, p=2, dim=-1, eps=1e-8) * F.normalize(v2, p=2, dim=-1, eps=1e-8), dim=-1).clamp(-1.0 + 1e-7,
                                                                                                       1.0 - 1e-7)
            L_angle = loss_bond_angle(torch.acos(cos), a_data['values'])  # 对齐 combine.py 的 values
        else:
            L_angle = torch.tensor(0.0, device=device)

        # Dihedral
        d_data = phy.get('dihedral', {})
        if d_data and d_data.get('indices') is not None and d_data['indices'].numel() > 0:
            idx = d_data['indices']
            p0, p1, p2, p3 = pred_coords[idx[:, 0]], pred_coords[idx[:, 1]], pred_coords[idx[:, 2]], pred_coords[
                idx[:, 3]]
            b0, b1, b2 = -(p1 - p0), p2 - p1, p3 - p2
            b1_n = F.normalize(b1, p=2, dim=-1, eps=1e-8)
            n1 = F.normalize(torch.cross(b0, b1_n), p=2, dim=-1, eps=1e-8)
            n2 = F.normalize(torch.cross(b1_n, b2), p=2, dim=-1, eps=1e-8)
            m1 = torch.cross(n1, b1_n)
            L_dihedral = loss_dihedral_angle(torch.atan2(torch.sum(m1 * n2, dim=-1), torch.sum(n1 * n2, dim=-1)),
                                             d_data['values'])  # 对齐 combine.py 的 values
        else:
            L_dihedral = torch.tensor(0.0, device=device)

        # VdW Repulsion
        v_data = phy.get('vdw', {})
        L_vdw = loss_vdw_repulsion(pred_coords, v_data['indices'], v_data['values'],
                                   self.params.get('vdw', {})) if (
                    v_data and v_data.get('indices') is not None) else torch.tensor(0.0, device=device)

        # Electrostatic
        dipoles = phy.get('dipole_vectors')
        e_data = phy.get('electro', {})
        L_electro = loss_electrostatic_dipole(pred_coords, dipoles, e_data.get('indices'),
                                               self.params['electrostatic']) if (
                    e_data and dipoles is not None) else torch.tensor(0.0, device=device)

        # HBond
        h_data = phy.get('hbond', {})
        L_hbond = loss_hydrogen_bond(pred_coords, h_data.get('indices'), self.params['hbond']) if (
                    h_data and h_data.get('indices') is not None) else torch.tensor(0.0, device=device)

        # Pi-Pi
        p_idx = phy.get('pi_pi', [])
        L_pi_pi = loss_pi_pi_stacking(_calculate_pi_pi_geometry(pred_coords, p_idx),
                                      self.params['pi_pi']) if p_idx else torch.tensor(0.0, device=device)

        L_bond = self._clean_component('L_bond', L_bond)
        L_angle = self._clean_component('L_angle', L_angle)
        L_dihedral = self._clean_component('L_dihedral', L_dihedral)
        L_vdw = self._clean_component('L_vdW', L_vdw)
        L_electro = self._clean_component('L_electro', L_electro)
        L_hbond = self._clean_component('L_hbond', L_hbond)
        L_pi_pi = self._clean_component('L_pi_pi', L_pi_pi)
        L_phy_geo = L_bond + L_angle + L_dihedral + L_vdw + L_electro + L_hbond + L_pi_pi
        L_phy_geo = self._clean_component('L_phy_geo', L_phy_geo)
        L_total = (w_noise * L_noise) + (w_struct * L_structure) + (w_phy * L_phy_geo)

        return {
            'total_loss': L_total, 'L_noise': L_noise, 'L_structure': L_structure, 'L_phy_geo': L_phy_geo,
            'L_bond': L_bond, 'L_angle': L_angle, 'L_dihedral': L_dihedral, 'L_vdW': L_vdw,
            'L_electro': L_electro, 'L_hbond': L_hbond, 'L_pi_pi': L_pi_pi
        }
