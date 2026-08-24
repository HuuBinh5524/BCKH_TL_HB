import torch

def dice_loss(pred_logits, target, smooth=1e-5):
    """
    Tính Dice Loss cho cả batch.
    pred_logits: Tensor (B, 1, H, W) - chưa qua Sigmoid
    target: Tensor (B, 1, H, W) - Ground Truth nhị phân (0 hoặc 1)
    """
    pred_soft = torch.sigmoid(pred_logits)
    intersection = (pred_soft * target).sum(dim=(1, 2, 3))
    union = pred_soft.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    loss = 1.0 - ((2.0 * intersection + smooth) / (union + smooth))
    return loss.mean()


def calculate_batch_metrics(pred_logits, target, smooth=1e-5):
    """
    Tính tổng các chỉ số Dice, IoU, Precision, Recall trên từng mẫu trong batch.
    Trả về: (dice_sum, iou_sum, precision_sum, recall_sum)
    """
    pred_bin = (torch.sigmoid(pred_logits) > 0.5).float()
    
    tp = (pred_bin * target).sum(dim=(1, 2, 3))
    fp = (pred_bin * (1.0 - target)).sum(dim=(1, 2, 3))
    fn = ((1.0 - pred_bin) * target).sum(dim=(1, 2, 3))

    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    return dice.sum().item(), iou.sum().item(), precision.sum().item(), recall.sum().item()


def calculate_cbl(pred_logits, target, smooth=1e-5):
    """
    Tính Center-Based Localization (CBL) score được tối ưu hóa trên GPU.
    Trả về: (tổng cbl_score, số mẫu hợp lệ trong batch)
    """
    pred_bin = (torch.sigmoid(pred_logits) > 0.5).float()
    B, _, H, W = pred_logits.shape

    ys = torch.arange(H, device=pred_logits.device, dtype=torch.float32)
    xs = torch.arange(W, device=pred_logits.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

    gt_area = target.sum(dim=(1, 2, 3))
    valid_mask = gt_area > smooth
    if not valid_mask.any():
        return 0.0, 0

    # Tính Trọng tâm (Centroid) GT và Prediction
    cx_gt = (grid_x * target).sum(dim=(1, 2, 3)) / (gt_area + smooth)
    cy_gt = (grid_y * target).sum(dim=(1, 2, 3)) / (gt_area + smooth)

    pred_area = pred_bin.sum(dim=(1, 2, 3))
    cx_p = (grid_x * pred_bin).sum(dim=(1, 2, 3)) / (pred_area + smooth)
    cy_p = (grid_y * pred_bin).sum(dim=(1, 2, 3)) / (pred_area + smooth)

    # Ước tính đường chéo Bounding Box GT batch-wise
    y_idx = grid_y.unsqueeze(0) * target.squeeze(1)
    x_idx = grid_x.unsqueeze(0) * target.squeeze(1)
    
    y_min = torch.where(target.squeeze(1) > 0, y_idx, torch.tensor(float('inf'), device=pred_logits.device)).flatten(1).min(dim=1)[0]
    y_max = torch.where(target.squeeze(1) > 0, y_idx, torch.tensor(float('-inf'), device=pred_logits.device)).flatten(1).max(dim=1)[0]
    x_min = torch.where(target.squeeze(1) > 0, x_idx, torch.tensor(float('inf'), device=pred_logits.device)).flatten(1).min(dim=1)[0]
    x_max = torch.where(target.squeeze(1) > 0, x_idx, torch.tensor(float('-inf'), device=pred_logits.device)).flatten(1).max(dim=1)[0]

    gt_diag = torch.sqrt((y_max - y_min) ** 2 + (x_max - x_min) ** 2) + smooth

    # Khoảng cách giữa 2 trọng tâm
    dist = torch.sqrt((cx_p - cx_gt) ** 2 + (cy_p - cy_gt) ** 2)
    cbl = torch.clamp(1.0 - dist / gt_diag, min=0.0)
    
    # Gán score = 0 với mẫu không dự đoán ra gì
    cbl = torch.where(pred_area < smooth, torch.tensor(0.0, device=pred_logits.device), cbl)
    
    return cbl[valid_mask].sum().item(), valid_mask.sum().item()