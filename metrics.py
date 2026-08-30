import torch
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

def dice_loss(pred, target, smooth=1e-5):
    """
    Tính Dice Loss cho cả batch.
    pred_logits: Tensor (B, 1, H, W) - chưa qua Sigmoid
    target: Tensor (B, 1, H, W) - Ground Truth nhị phân (0 hoặc 1)
    """
    pred_soft = torch.sigmoid(pred)
    intersection = (pred_soft * target).sum(dim=(1, 2, 3))
    union = pred_soft.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    loss = 1.0 - ((2.0 * intersection + smooth) / (union + smooth))
    return loss.mean()


def calculate_batch_metrics(pred, target, smooth=1e-5):
    """
    Tính tổng các chỉ số Dice, IoU, Precision, Recall trên từng mẫu trong batch.
    Trả về: (dice_sum, iou_sum, precision_sum, recall_sum)
    """
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    
    tp = (pred_bin * target).sum(dim=(1, 2, 3))
    fp = (pred_bin * (1.0 - target)).sum(dim=(1, 2, 3))
    fn = ((1.0 - pred_bin) * target).sum(dim=(1, 2, 3))

    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)

    return dice.sum().item(), iou.sum().item(), precision.sum().item(), recall.sum().item()


def compute_hd95(pred_mask, gt_mask):
    """
    Compute symmetric 95th percentile Hausdorff Distance (HD95)
    between two binary masks.

    Args:
        pred_mask: numpy array, shape (H, W), binary
        gt_mask:   numpy array, shape (H, W), binary

    Returns:
        float: HD95 in pixels
    """
    pred_mask = np.asarray(pred_mask, dtype=bool)
    gt_mask = np.asarray(gt_mask, dtype=bool)

    # Both masks are empty -> perfect agreement.
    if not pred_mask.any() and not gt_mask.any():
        return 0.0

    # One mask is empty -> HD95 is undefined.
    # Return NaN so it can be excluded from the mean.
    if not pred_mask.any() or not gt_mask.any():
        return np.nan

    # Extract object boundaries.
    pred_boundary = pred_mask ^ binary_erosion(pred_mask)
    gt_boundary = gt_mask ^ binary_erosion(gt_mask)

    # Distance from every pixel to the opposite boundary.
    dist_to_gt = distance_transform_edt(~gt_boundary)
    dist_to_pred = distance_transform_edt(~pred_boundary)

    pred_to_gt = dist_to_gt[pred_boundary]
    gt_to_pred = dist_to_pred[gt_boundary]

    # Symmetric 95th percentile Hausdorff distance.
    hd95 = max(
        np.percentile(pred_to_gt, 95),
        np.percentile(gt_to_pred, 95)
    )

    return float(hd95)


def calculate_cbl(pred, target, smooth=1e-6):
    """
    CBL, Center-Based Localization score in [0, 1].
    Measures how close the predicted mask's centroid is to the GT mask's centroid.
    Normalized by the GT bbox diagonal to stay scale-invariant.
    Returns (cbl_sum, valid_count).
    """
    B, _, H, W = pred.shape
    pred_bin = (torch.sigmoid(pred) > 0.5).float()

    ys = torch.arange(H, device=pred.device, dtype=torch.float32)
    xs = torch.arange(W, device=pred.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W)

    cbl_sum, valid_count = 0.0, 0

    for b in range(B):
        gt_m   = target[b, 0]
        pred_m = pred_bin[b, 0]
        gt_area = gt_m.sum()

        if gt_area < smooth:
            continue  # empty GT, skip

        # GT centroid
        cx_gt = (grid_x * gt_m).sum() / (gt_area + smooth)
        cy_gt = (grid_y * gt_m).sum() / (gt_area + smooth)

        # GT bbox diagonal
        nz    = gt_m.nonzero()
        gt_diag = torch.sqrt(
            ((nz[:, 0].max() - nz[:, 0].min()).float()) ** 2 +
            ((nz[:, 1].max() - nz[:, 1].min()).float()) ** 2
        ) + smooth

        pred_area = pred_m.sum()
        if pred_area < smooth:
            valid_count += 1  # CBL = 0 for this sample
            continue

        # Predicted mask centroid
        cx_p = (grid_x * pred_m).sum() / (pred_area + smooth)
        cy_p = (grid_y * pred_m).sum() / (pred_area + smooth)

        d   = torch.sqrt((cx_p - cx_gt) ** 2 + (cy_p - cy_gt) ** 2)
        cbl = torch.clamp(1.0 - d / gt_diag, min=0.0)
        cbl_sum += cbl.item()
        valid_count += 1

    return cbl_sum, valid_count


def compute_image_level_metrics(image_groups, threshold=0.5):
    """
    Compute Dice, IoU, Precision, Recall, HD95 and CBL
    at IMAGE LEVEL.

    image_groups:
        {
            image_id: {
                'preds': [Tensor(1,1,H,W), ...],
                'gts':   [Tensor(1,1,H,W), ...]
            }
        }

    Each polygon/GT is first predicted independently.
    Predictions and GTs belonging to the same source image
    are then merged before computing metrics.

    Returns:
        dict containing image-level mean metrics.
    """

    metric_values = {
        'dice': [],
        'iou': [],
        'precision': [],
        'recall': [],
        'hd95': [],
        'cbl': []
    }

    for image_id, data in image_groups.items():
        # 1. Merge all polygon predictions of this image
        pred_stack = torch.stack(data['preds'], dim=0)
        combined_pred_prob = torch.max(pred_stack, dim=0)[0]

        combined_pred = (
            combined_pred_prob > threshold
        ).float()

        # 2. Merge all GT polygons of this image
        gt_stack = torch.stack(data['gts'], dim=0)
        combined_gt = torch.max(gt_stack, dim=0)[0]

        combined_gt = (
            combined_gt > 0.5
        ).float()

        # Convert to NumPy for image-level evaluation
        pred_np = combined_pred[0, 0].cpu().numpy().astype(bool)
        gt_np = combined_gt[0, 0].cpu().numpy().astype(bool)

        # 3. Dice / IoU / Precision / Recall
        tp = np.logical_and(pred_np, gt_np).sum()
        fp = np.logical_and(pred_np, ~gt_np).sum()
        fn = np.logical_and(~pred_np, gt_np).sum()

        smooth = 1e-5

        dice = (
            2.0 * tp + smooth
        ) / (
            2.0 * tp + fp + fn + smooth
        )

        iou = (
            tp + smooth
        ) / (
            tp + fp + fn + smooth
        )

        precision = (
            tp + smooth
        ) / (
            tp + fp + smooth
        )

        recall = (
            tp + smooth
        ) / (
            tp + fn + smooth
        )

        metric_values['dice'].append(float(dice))
        metric_values['iou'].append(float(iou))
        metric_values['precision'].append(float(precision))
        metric_values['recall'].append(float(recall))

        # 4. HD95
        hd95 = compute_hd95(pred_np, gt_np)

        if not np.isnan(hd95):
            metric_values['hd95'].append(hd95)

        # 5. CBL
        cbl_sum, cbl_count = calculate_cbl(
            combined_pred,
            combined_gt
        )

        if cbl_count > 0:
            metric_values['cbl'].append(
                cbl_sum / cbl_count
            )

    # 6. Mean across IMAGES, not polygons
    results = {
        'num_images': len(image_groups),
        'dice': (
            float(np.mean(metric_values['dice']))
            if metric_values['dice'] else 0.0
        ),
        'iou': (
            float(np.mean(metric_values['iou']))
            if metric_values['iou'] else 0.0
        ),
        'precision': (
            float(np.mean(metric_values['precision']))
            if metric_values['precision'] else 0.0
        ),
        'recall': (
            float(np.mean(metric_values['recall']))
            if metric_values['recall'] else 0.0
        ),
        'hd95': (
            float(np.mean(metric_values['hd95']))
            if metric_values['hd95'] else float('nan')
        ),
        'cbl': (
            float(np.mean(metric_values['cbl']))
            if metric_values['cbl'] else 0.0
        )
    }
    return results