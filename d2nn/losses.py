import torch
import torch.nn.functional as F

def field_energy(field: torch.Tensor) -> torch.Tensor:
    return (torch.abs(field) ** 2).flatten(1).sum(dim=1)

def pcc(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    am = a.mean(dim=1, keepdim=True)
    bm = b.mean(dim=1, keepdim=True)
    ac = a - am
    bc = b - bm
    num   = (ac * bc).sum(dim=1)
    denom = torch.sqrt((ac ** 2).sum(dim=1) * (bc ** 2).sum(dim=1)) + eps
    return num / denom

def normalize_input_to_canvas_energy(x: torch.Tensor, target_l2_sq: float) -> torch.Tensor:
    e = field_energy(x)
    scale = torch.sqrt(target_l2_sq / (e + 1e-8))
    return x * scale.unsqueeze(1).unsqueeze(2)

def scale_target_to_input_energy(target: torch.Tensor,
                                 input_l2_sq: torch.Tensor) -> torch.Tensor:
    e = field_energy(target)
    scale = torch.sqrt(input_l2_sq / (e + 1e-8))
    return target * scale.unsqueeze(1).unsqueeze(2)

def target_loss(out_field: torch.Tensor,
                labels: torch.Tensor,
                mask_pos: torch.Tensor,
                mask_neg: torch.Tensor,
                ce_scale: float = 100.0,
                alpha_ce: float = 1.0,
                alpha_anti_neg: float = 2.0,
                alpha_bg: float = 5.0) -> torch.Tensor:
    I_out = torch.abs(out_field) ** 2

    total_energy = I_out.flatten(start_dim=1).sum(dim=1) + 1e-8

    E_pos_all = torch.stack([(I_out * mask_pos[k]).flatten(1).sum(dim=1) for k in range(mask_pos.shape[0])], dim=1)
    E_neg_all = torch.stack([(I_out * mask_neg[k]).flatten(1).sum(dim=1) for k in range(mask_neg.shape[0])], dim=1)
    S = E_pos_all - E_neg_all
    L_ce = F.cross_entropy(S * ce_scale, labels)

    current_pos_mask = mask_pos[labels]
    current_neg_mask = mask_neg[labels]

    L_anti_neg = torch.tensor(0.0, device=out_field.device)
    if alpha_anti_neg > 0.0:

        target_neg_energy = (I_out * current_neg_mask).flatten(1).sum(dim=1)

        L_anti_neg = (target_neg_energy / total_energy).mean()

    L_bg = torch.tensor(0.0, device=out_field.device)
    if alpha_bg > 0.0:

        bg_mask = torch.clamp(1.0 - current_pos_mask - current_neg_mask, min=0.0)

        bg_energy = (I_out * bg_mask).flatten(1).sum(dim=1)

        L_bg = (bg_energy / total_energy).mean()

    total_loss = (alpha_ce * L_ce) + \
                 (alpha_anti_neg * L_anti_neg) + \
                 (alpha_bg * L_bg)

    return total_loss
