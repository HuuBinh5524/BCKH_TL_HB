import os
import random
import logging
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PromptSegmentationDataset
from models.networks.pga_unet_2D import PGA_UNet
from metrics import dice_loss, calculate_batch_metrics, calculate_cbl


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# ARGUMENT PARSER
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="PGA-UNet Training Pipeline")
    
    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training and val")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--early_stop", type=int, default=15, help="Patience for early stopping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--img_size", type=int, default=512, help="Input image resolution")

    # Mode & Model Configuration
    parser.add_argument("--model_name", type=str, default="pga", help="Name of model to use")
    parser.add_argument("--train_prompt_mode", type=str, default="random", choices=["zoom_out", "shift"],
                        help="Prompt mode for training")
    parser.add_argument("--dataset_path", type=str, default="", help="Path of the dataset")    
    parser.add_argument("--dataset_name", type=str, default="BTXRD", help="Name of the dataset to use")

    return parser.parse_args()

# =========================================================
# LOGGER FUNCTION
# =========================================================
def setup_logger(exp_name):
    os.makedirs("logs", exist_ok=True)
    t = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(f"logs/train_exp_{exp_name}_{t}.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

# =========================================================
# MAIN FUNCTION
# =========================================================
def main():
    args = parse_args()
    logger = setup_logger(args.train_prompt_mode)

    # ── Dataset ──────────────────────────────────────────────────────
    ## Train Data
    train_ds = PromptSegmentationDataset(
        image_dir=os.path.join(args.dataset_path, "train", "images"),
        json_dir=os.path.join(args.dataset_path, "train", "annotations"),
        img_size=args.img_size, is_train=True,
        prompt_mode=args.train_prompt_mode
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                num_workers=2, pin_memory=True)

    ## Val Data
    val_ds = PromptSegmentationDataset(
        image_dir=os.path.join(args.dataset_path, "val", "images"),
        json_dir=os.path.join(args.dataset_path, "val", "annotations"),
        img_size=args.img_size, is_train=False,
        prompt_mode=args.train_prompt_mode
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────
    model = PGA_UNet(in_channels=1, n_classes=1,
                        use_encoder_prompt=True).to(device)
    criterion_bce = nn.BCEWithLogitsLoss()
    optimizer     = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler     = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)

    os.makedirs("checkpoints", exist_ok=True)
    best_val_dice   = 0.0
    patience_counter = 0
    ckpt_prefix     = f"checkpoints/pga_unet_{args.img_size}"

    # ── Epoch Loop ───────────────────────────────────────────────────
    for epoch in range(args.epochs):
        # 1. Train Phase
        model.train()
        train_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")

        for images, masks, prompts in train_loop:
            images, masks, prompts = (images.to(device), masks.to(device), prompts.to(device))
            preds = model(images, prompts)
            loss  = criterion_bce(preds, masks) + dice_loss(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            train_loop.set_postfix(loss=f"{loss.item():.4f}")

        train_loss_avg = train_loss / len(train_loader)

        # 2. Validate Phase
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou  = 0.0
        val_pre  = 0.0
        val_rec  = 0.0

        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]", leave=False)

        with torch.no_grad():
            for images, masks, prompts in val_loop:
                images, masks, prompts = images.to(device), masks.to(device), prompts.to(device)

                # Forward pass
                preds = model(images, prompts)

                # Tính Val Loss
                v_loss = criterion_bce(preds, masks) + dice_loss(preds, masks)
                val_loss += v_loss.item()

                # Tính Metrics
                dice, iou, pre, rec = calculate_batch_metrics(preds, masks)
                val_dice += dice
                val_iou  += iou
                val_pre  += pre
                val_rec  += rec

                val_loop.set_postfix({
                    'loss': f"{v_loss.item():.4f}",
                    'dice': f"{dice:.4f}",
                    'iou': f"{iou:.4f}"
                })

        num_val_batches = len(val_loader)
        val_results = {
            'loss': val_loss / num_val_batches,
            'dice': val_dice / num_val_batches,
            'iou': val_iou / num_val_batches,
            'precision': val_pre / num_val_batches,
            'recall': val_rec / num_val_batches
        }

        # 3. Learning Rate Scheduler & Logging
        primary_dice = val_results['dice']
        scheduler.step(primary_dice)

        log_str = (f"Epoch {epoch+1:03d}/{args.epochs} | "
                   f"Train_loss: {train_loss_avg:.4f} | "
                   f"Val_loss: {val_results['loss']:.4f} | "
                   f"Dice: {val_results['dice']:.4f} | "
                   f"IoU: {val_results['iou']:.4f} | "
                   f"Prec: {val_results['precision']:.4f} | "
                   f"Rec: {val_results['recall']:.4f} | "
                   f"LR: {optimizer.param_groups[0]['lr']:.1e}")

        # Save last checkpoint
        torch.save(model.state_dict(), f"{ckpt_prefix}_last.pth")

        # Check Best Checkpoint
        if primary_dice > best_val_dice:
            best_val_dice = primary_dice
            torch.save(model.state_dict(), f"{ckpt_prefix}_best.pth")
            log_str = "🥇 [BEST] " + log_str
            patience_counter = 0
        else:
            patience_counter += 1

        logger.info(log_str)

        # Early Stopping
        if patience_counter >= args.early_stop:
            logger.info(f"Early stopping triggered at epoch {epoch+1}.")
            break

    logger.info(f"\nTraining Complete!")
    logger.info(f"Best Val Dice ({args.train_prompt_mode}): {best_val_dice:.4f}")
    logger.info(f"Best Checkpoint: {ckpt_prefix}_best.pth")

if __name__ == "__main__":
    main()