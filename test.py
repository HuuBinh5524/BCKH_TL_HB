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
from metrics import compute_image_level_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# ARGUMENT PARSER
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="PGA-UNet Image-Level Evaluation Pipeline")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        default="checkpoints/pga_unet_512_best.pth", help="Path to trained model checkpoint (.pth)")
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
    logger.info(f"🚀 Bắt đầu đánh giá mô hình (Cấp độ ảnh - Image Level)")
    logger.info(f"📂 Checkpoint: {args.checkpoint}")
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
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
    
    # Hỗ trợ tải checkpoint dù lưu dưới dạng dict hay state_dict trực tiếp
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    # 3. Tiến hành suy luận theo cấp ảnh (Image-level Grouping)
    image_groups = defaultdict(lambda: {'preds': [], 'gts': []})

    with torch.no_grad():
        for batch_idx, (images, masks, prompts) in enumerate(tqdm(test_loader, desc="[Inference]")):
            images = images.to(DEVICE)
            prompts = prompts.to(DEVICE)
            masks = masks.to(DEVICE)

            preds_logits = model(images, prompts)
            preds_prob = torch.sigmoid(preds_logits)

            batch_sz = images.size(0)

            for i in range(batch_sz):
                sample_idx = batch_idx * test_loader.batch_size + i
                img_id, shape_idx = test_ds.all_samples[sample_idx]

                # Lưu prediction của polygon này
                image_groups[img_id]['preds'].append(
                    preds_prob[i:i+1].cpu()
                )

                # Lưu GT của polygon này
                image_groups[img_id]['gts'].append(
                    masks[i:i+1].cpu()
                )

    # 4. Đánh giá theo Cấp độ Ảnh
    results = compute_image_level_metrics(
        image_groups,
        threshold=args.threshold
    )

    logger.info("\n" + "=" * 60)
    logger.info(f"🏆 KẾT QUẢ ĐÁNH GIÁ CẤP ĐỘ ẢNH ({results['num_images']} ẢNH)")
    logger.info("=" * 60)

    logger.info(f"🎯 Dice            : {results['dice']:.4f}")
    logger.info(f"📐 IoU             : {results['iou']:.4f}")
    logger.info(f"🔍 Precision       : {results['precision']:.4f}")
    logger.info(f"📢 Recall          : {results['recall']:.4f}")
    logger.info(f"📏 HD95            : {results['hd95']:.4f} pixels")
    logger.info(f"🎯 CBL             : {results['cbl']:.4f}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()