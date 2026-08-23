import os
import argparse
import logging
import datetime
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from dataset import PromptSegmentationDataset
from models.networks.pga_unet_2D import PGA_UNet
from metrics import calculate_batch_metrics, calculate_cbl

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# ARGUMENT PARSER
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="PGA-UNet Image-Level Evaluation Pipeline")
    
    parser.add_argument("--Checkpoint", type=str, required=True
                        default="checkpoints/pga_unet_512.pth", help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset root folder")
    parser.add_argument("--dataset_name", type=str, default="BTXRD", help="Dataset name")
    parser.add_argument("--img_size", type=int, default=512, help="Input image resolution")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for testing")
    parser.add_argument("--prompt_mode", type=str, default="zoom_out", choices=["zoom_out", "shift"],
                        help="Prompt mode for testing")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold for predictions")
    parser.add_argument("--save_preds", action="store_true", help="Save combined prediction masks to disk")
    parser.add_argument("--output_dir", type=str, default="test_results", help="Directory to save test outputs")

    return parser.parse_args()

# =========================================================
# LOGGER SETUP
# =========================================================
def setup_logger(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    t = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"eval_log_{t}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

# =========================================================
# MAIN EVALUATION FUNCTION
# =========================================================
def main():
    args = parse_args()
    logger = setup_logger(args.output_dir)

    logger.info("=" * 60)
    logger.info(f"🚀 Bắt đầu đánh giá mô hình PGA-UNet (Cấp độ ảnh - Image Level)")
    logger.info(f"📂 Checkpoint: {args.Checkpoint}")
    logger.info(f"📂 Dataset Path: {args.dataset_path}")
    logger.info(f"🎯 Prompt Mode: {args.prompt_mode} | Threshold: {args.threshold}")
    logger.info("=" * 60)

    # 1. Khởi tạo Dataset & DataLoader
    test_ds = PromptSegmentationDataset(
        image_dir=os.path.join(args.dataset_path, "test", "images"),
        json_dir=os.path.join(args.dataset_path, "test", "annotations"),
        img_size=args.img_size,
        is_train=False,
        prompt_mode=args.prompt_mode
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    # 2. Khởi tạo & Tải Trọng số Mô hình
    model = PGA_UNet(in_channels=1, n_classes=1, use_encoder_prompt=True).to(DEVICE)
    checkpoint = torch.load(args.weights, map_location=DEVICE)
    
    # Hỗ trợ tải checkpoint dù lưu dưới dạng dict hay state_dict trực tiếp
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    # 3. Tiến hành suy luận và Gom nhóm theo Ảnh (Image-level Grouping)
    # Cấu trúc image_groups: { image_id: {'preds': [tensor, ...], 'gts': [tensor, ...]} }
    image_groups = defaultdict(lambda: {'preds': [], 'gts': []})

    logger.info("\n🔄 Đang chạy Inference trên tập test...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="[Inference]")):
            # Xử lý linh hoạt dữ liệu trả về từ DataLoader
            if len(batch) == 4:
                images, masks, prompts, img_ids = batch
            else:
                images, masks, prompts = batch[:3]
                img_ids = None

            images = images.to(DEVICE)
            prompts = prompts.to(DEVICE)
            masks = masks.to(DEVICE)

            # Raw logits từ mô hình -> Sigmoid -> Probability
            preds_logits = model(images, prompts)
            preds_prob = torch.sigmoid(preds_logits)

            batch_sz = images.size(0)
            for i in range(batch_sz):
                # Xác định định danh ảnh (Image ID / Filename)
                if img_ids is not None:
                    img_id = img_ids[i]
                else:
                    sample_idx = batch_idx * test_loader.batch_size + i
                    if hasattr(test_ds, 'samples') and 'image_name' in test_ds.samples[sample_idx]:
                        img_id = test_ds.samples[sample_idx]['image_name']
                    elif hasattr(test_ds, 'image_files'):
                        img_id = test_ds.image_files[sample_idx]
                    else:
                        img_id = f"image_{sample_idx:04d}"

                # Lưu trữ kết quả từng prompt/tổn thương của ảnh tương ứng
                image_groups[img_id]['preds'].append(preds_prob[i:i+1]) # Keep shape [1, C, H, W]
                image_groups[img_id]['gts'].append(masks[i:i+1])

    logger.info(f"✅ Hoàn thành suy luận. Tổng số ảnh độc lập: {len(image_groups)}")

    # 4. Ghép các vùng tổn thương & Đánh giá theo Cấp độ Ảnh
    logger.info("\n📊 Đang ghép mask & Tính toán metrics cấp độ ảnh...")

    all_dice = []
    all_iou = []
    all_precision = []
    all_recall = []
    all_cbl = []

    if args.save_preds:
        save_dir = os.path.join(args.output_dir, "predicted_masks")
        os.makedirs(save_dir, exist_ok=True)

    for img_id, data in tqdm(image_groups.items(), desc="[Image-Level Eval]"):
        # Ghép tất cả dự đoán của 1 ảnh thành 1 mask duy nhất (Logical OR qua max probability)
        combined_pred_prob = torch.max(torch.stack(data['preds'], dim=0), dim=0)[0]
        # Binarize dự đoán theo threshold
        combined_pred_bin = (combined_pred_prob > args.threshold).float()

        # Ghép tất cả GT mask của 1 ảnh thành 1 GT mask duy nhất
        combined_gt_bin = torch.max(torch.stack(data['gts'], dim=0), dim=0)[0]
        combined_gt_bin = (combined_gt_bin > 0.5).float()

        # Tính toán các chỉ số cơ bản từ metrics.py cho bức ảnh hợp nhất này
        dice, iou, pre, rec = calculate_batch_metrics(combined_pred_bin, combined_gt_bin)

        all_dice.append(dice)
        all_iou.append(iou)
        all_precision.append(pre)
        all_recall.append(rec)

        # Tính chỉ số CBL (Boundary Loss / Contour-based evaluation) nếu có
        try:
            cbl_val = calculate_cbl(combined_pred_bin, combined_gt_bin)
            if isinstance(cbl_val, torch.Tensor):
                cbl_val = cbl_val.item()
            all_cbl.append(cbl_val)
        except Exception:
            pass  # Bỏ qua nếu hàm calculate_cbl yêu cầu định dạng đầu vào khác

        # Lưu ảnh mask ghép nếu có yêu cầu
        if args.save_preds:
            mask_np = (combined_pred_bin.squeeze().cpu().numpy() * 255).astype(np.uint8)
            save_path = os.path.join(save_dir, f"{os.path.splitext(img_id)[0]}_pred.png")
            Image.fromarray(mask_np).save(save_path)

    # 5. Tính trung bình và In kết quả tổng hợp
    mean_dice = np.mean(all_dice)
    mean_iou = np.mean(all_iou)
    mean_precision = np.mean(all_precision)
    mean_recall = np.mean(all_recall)
    mean_cbl = np.mean(all_cbl) if len(all_cbl) > 0 else None

    logger.info("\n" + "=" * 60)
    logger.info("🏆 KẾT QUẢ ĐÁNH GIÁ CẤP ĐỘ ẢNH (IMAGE-LEVEL RESULTS)")
    logger.info("=" * 60)
    logger.info(f"📸 Tổng số ảnh evaluated : {len(image_groups)}")
    logger.info(f"🎯 Mean Dice Score      : {mean_dice:.4f}")
    logger.info(f"📐 Mean IoU             : {mean_iou:.4f}")
    logger.info(f"🔍 Mean Precision       : {mean_precision:.4f}")
    logger.info(f"📢 Mean Recall          : {mean_recall:.4f}")
    if mean_cbl is not None:
        logger.info(f"🌊 Mean CBL             : {mean_cbl:.4f}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()