import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import gc
import random
import numpy as np
import argparse
import utils
import visualize_char
from HAT import HATCharClassifier
# from data_loading_char import StrokeSegmentDataset, custom_collate_fn
from data_loading_isi import ISISegmentDataset, custom_collate_fn
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(
    model, train_loader, val_loader, device, label_converter, logger, writer, save_path,
    num_epochs=10, lr=1e-4, visualize_every=10, early_stop_patience=5
):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 120)
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        skipped_batches = 0

        for batch_idx, (strokes, images, labels, stroke_masks) in enumerate(tqdm(train_loader, desc="Training")):
            try:
                strokes = strokes.to(device) if strokes is not None else None
                images = images.to(device) if images is not None else None
                stroke_masks = stroke_masks.to(device) if stroke_masks is not None else None

                labels = label_converter.encode(labels).to(device)  # (B,)

                optimizer.zero_grad()
                logits, stroke_features = model(images, strokes, stroke_masks)  # (B, vocab_size)

                # Visualization (every N batches)
                if batch_idx % visualize_every == 0:
                    visualize_char.visualize_char_classifier(
                        images, logits, stroke_features, labels, label_converter,
                        epoch, batch_idx, writer, save_path=save_path, phase="train", max_samples=10
                    )
                loss = ce_criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            except RuntimeError as e:
                print(f"\nSkipping batch number: {batch_idx} due to error: {e}")
                torch.cuda.empty_cache()
                skipped_batches += 1
                continue

        train_loss /= len(train_loader)
        train_acc = correct / total
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Skipped {skipped_batches} batches in epoch {epoch + 1}")
        logger.info(f'\nEpoch : {epoch + 1} \t training loss : {train_loss:0.5f} \t training acc : {train_acc:0.5f}')
        logger.info(f"Skipped {skipped_batches} batches in epoch {epoch + 1}")
        writer.add_scalar("train/loss", train_loss, epoch + 1)
        writer.add_scalar("train/acc", train_acc, epoch + 1)
        torch.cuda.empty_cache()

        # Validation phase (with accuracy and additional metrics)
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        skipped_val_batches = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, (strokes, images, labels, stroke_masks) in enumerate(tqdm(val_loader, desc="Validation")):
                try:
                    strokes = strokes.to(device) if strokes is not None else None
                    images = images.to(device) if images is not None else None
                    stroke_masks = stroke_masks.to(device) if stroke_masks is not None else None

                    labels = label_converter.encode(labels).to(device)
                    logits, stroke_features = model(images, strokes, stroke_masks)

                    # Visualization (every N batches)
                    if batch_idx % visualize_every == 0:
                        visualize_char.visualize_char_classifier(
                            images, logits, stroke_features, labels, label_converter,
                            epoch, batch_idx, writer, save_path=save_path, phase="val", max_samples=10, 
                        )
                    loss = ce_criterion(logits, labels)
                    val_loss += loss.item()
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                except RuntimeError as e:
                    print(f"Skipping batch number : {batch_idx} due to error: {e}")
                    torch.cuda.empty_cache()
                    skipped_val_batches += 1
                    continue

            val_loss /= len(val_loader)
            val_acc = correct / total

            # Additional metrics
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            cm = confusion_matrix(all_labels, all_preds)

            print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            print(f"Skipped {skipped_val_batches} validation batches in epoch {epoch + 1}")

            writer.add_scalar("val/loss", val_loss, epoch + 1)
            writer.add_scalar("val/acc", val_acc, epoch + 1)
            writer.add_scalar("val/precision", precision, epoch + 1)
            writer.add_scalar("val/recall", recall, epoch + 1)
            writer.add_scalar("val/f1", f1, epoch + 1)
            writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch + 1)
            logger.info(f'Epoch : {epoch + 1} \t Validation Loss : {val_loss:0.5f} \t Validation Acc : {val_acc:0.5f}')
            logger.info(f'Epoch : {epoch + 1} \t Precision : {precision:0.5f} \t Recall : {recall:0.5f} \t F1 : {f1:0.5f}')
            logger.info(f"Skipped {skipped_val_batches} validation batches in epoch {epoch + 1}")

            
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(cm, ax=ax, cmap='Blues')
            ax.set_title('Confusion Matrix')
            writer.add_figure('val/confusion_matrix', fig, epoch + 1)
            plt.close(fig)

        scheduler.step()
        torch.cuda.empty_cache()

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            # torch.save({
            #     "epoch": epoch + 1,
            #     "model_state_dict": model.state_dict(),
            #     "optimizer_state_dict": optimizer.state_dict(),
            #     "loss": val_loss,
            #     "val_acc": val_acc
            # }, f"./{save_path}/best_model.pth")
            # print(f"Best model saved at epoch {epoch+1} with val_acc {val_acc:.4f}")
        else:
            epochs_no_improve += 1

        gc.collect()

        # Early stopping
        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    print("Training complete!")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default="./exp/vit")
    parser.add_argument('--train_data', type=str, default="./isi-air-dataset/train")
    parser.add_argument('--val_data', type=str, default="./isi-air-dataset/test")
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--val_batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--visualize_every', type=int, default=50)
    parser.add_argument('--early_stop_patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--input_mode', type=str, default="both", choices=["both", "image", "stroke"],
                    help="Specify input mode: 'both', 'image', or 'stroke'")
    parser.add_argument('--description', type=str, default="None")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    save_path = args.save_path
    writer = SummaryWriter(save_path)
    logger = utils.get_logger(save_path)
    # alphabet = ".,;:'/\"\\!?-()[]\{\}*#+ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" #iam
    # alphabet = "':;,!.-()[]#*+?\"/ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" #iam rebalanced
    # alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYabcdefghijklmnopqrstuvxyzÀÁĂÂÔÚàáâãèéêìíòóôõùúýăĐđĩũƠơƯưạẢảẤấẦầẩẫậắằẳẵặẹẻẽếỀềỂểễỆệỉịọỏỐốỒồổỗộớờỞởỡợụỦủứừửữựỳỷỹ"  #vnondb
    alphabet = "0123456789"
    label_converter = utils.CharLabelConverter(set(alphabet))
    vocab_size = len(set(alphabet))
    num_epochs = args.num_epochs
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size

    # Log all arguments
    logger.info("Arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    logger.info('Loading train loader...')
    # trainset = StrokeSegmentDataset(args.train_data, augmentation_probability=0.0, input_mode=args.input_mode)
    # train_loader = DataLoader(
    #     trainset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True,
    #     collate_fn=custom_collate_fn
    # )
    train_dataset = ISISegmentDataset(root_dir=args.train_data, input_mode=args.input_mode, use_opencv_aug=False, augmentation_probability=0.0)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True,
    collate_fn=custom_collate_fn)

    logger.info('Loading val loader...')
    # valset = StrokeSegmentDataset(args.val_data, augmentation_probability=0.0, input_mode=args.input_mode)
    # val_loader = DataLoader(
    #     valset, batch_size=val_batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True,
    #      collate_fn=custom_collate_fn
    # )
    val_dataset = ISISegmentDataset(root_dir=args.val_data, input_mode=args.input_mode, use_opencv_aug=False, augmentation_probability=0.0)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True,
    collate_fn=custom_collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HATCharClassifier(
    patch_dim=256,
    stroke_dim=3,
    model_dim=256,
    vocab_size=vocab_size,
    input_mode=args.input_mode
    )

    logger.info('{}'.format(model))
    total_param = sum(p.numel() for p in model.parameters())
    logger.info('total_param is {}'.format(total_param))

    train_model(
        model, train_loader, val_loader, device, label_converter, logger, writer, save_path=save_path, 
        num_epochs=num_epochs, lr=args.lr, visualize_every=args.visualize_every, early_stop_patience=args.early_stop_patience
    )

if __name__ == "__main__":
    main()