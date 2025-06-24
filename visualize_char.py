import os
import torch
import torchvision
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

def visualize_char_classifier(
    images, logits, stroke_features, labels, label_converter, epoch, batch_idx, writer,
    save_path, phase="train", max_samples=4
):
    def save_fig(fig, save_dir, name):
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{name}.png"))
        plt.close(fig)

    save_dir = f"{save_path}/{phase}/epoch_{epoch}"

    # 1. Input images (only if available)
    if images is not None:
        try:
            if images.shape[1] == 1:
                img_vis = images[:max_samples].repeat(1, 3, 1, 1)
            else:
                img_vis = images[:max_samples]
            img_grid = torchvision.utils.make_grid(img_vis.cpu(), nrow=2, normalize=True, scale_each=True)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(img_grid.permute(1, 2, 0).numpy())
            ax.axis('off')
            ax.set_title("Input Images")
            save_fig(fig, save_dir, "input_images")
            writer.add_image(f'{phase}/input_images', img_grid, epoch)
        except Exception as e:
            print(f"Image visualization failed: {e}")

    # 2. Predicted vs. Ground Truth text
    preds = logits.argmax(dim=-1)
    preds_str = label_converter.decode(preds)
    labels_str = label_converter.decode(labels)
    table_str = "| Sample | Predicted | Ground Truth |\n|---|---|---|\n"
    for i, (pred, gt) in enumerate(zip(preds_str[:max_samples], labels_str[:max_samples])):
        table_str += f"| {i} | {pred} | {gt} |\n"
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "pred_vs_gt.txt"), "w", encoding="utf-8") as f:
        f.write(table_str)
    writer.add_text(f'{phase}/pred_vs_gt', table_str, epoch)

    # 3–4. t-SNE & UMAP of logits
    try:
        logits_np = logits.detach().cpu().numpy()
        tsne = TSNE(n_components=2, random_state=42)
        logits_2d = tsne.fit_transform(logits_np)
        fig, ax = plt.subplots()
        scatter = ax.scatter(logits_2d[:, 0], logits_2d[:, 1], c=preds.cpu().numpy(), cmap='tab20', s=30)
        ax.set_title("t-SNE of Logits")
        plt.colorbar(scatter, ax=ax, label='Predicted Class')
        save_fig(fig, save_dir, "logits_tsne")
        writer.add_figure(f'{phase}/logits_tsne', fig, epoch)
    except Exception as e:
        print(f"t-SNE failed (logits): {e}")

    try:
        reducer = umap.UMAP(n_components=2, random_state=42)
        logits_umap = reducer.fit_transform(logits_np)
        fig, ax = plt.subplots()
        scatter = ax.scatter(logits_umap[:, 0], logits_umap[:, 1], c=preds.cpu().numpy(), cmap='tab20', s=30)
        ax.set_title("UMAP of Logits")
        plt.colorbar(scatter, ax=ax, label='Predicted Class')
        save_fig(fig, save_dir, "logits_umap")
        writer.add_figure(f'{phase}/logits_umap', fig, epoch)
    except Exception as e:
        print(f"UMAP failed (logits): {e}")

    # 5–6. t-SNE & UMAP of stroke features (only if available)
    if stroke_features is not None:
        try:
            stroke_np = stroke_features.detach().cpu().numpy()
            tsne = TSNE(n_components=2, random_state=42)
            stroke_2d = tsne.fit_transform(stroke_np)
            fig, ax = plt.subplots()
            scatter = ax.scatter(stroke_2d[:, 0], stroke_2d[:, 1], c=preds.cpu().numpy(), cmap='tab20', s=30)
            ax.set_title("t-SNE of Stroke Features")
            plt.colorbar(scatter, ax=ax, label='Predicted Class')
            save_fig(fig, save_dir, "stroke_tsne")
            writer.add_figure(f'{phase}/stroke_tsne', fig, epoch)
        except Exception as e:
            print(f"t-SNE failed (stroke features): {e}")

        try:
            reducer = umap.UMAP(n_components=2, random_state=42)
            stroke_umap = reducer.fit_transform(stroke_np)
            fig, ax = plt.subplots()
            scatter = ax.scatter(stroke_umap[:, 0], stroke_umap[:, 1], c=preds.cpu().numpy(), cmap='tab20', s=30)
            ax.set_title("UMAP of Stroke Features")
            plt.colorbar(scatter, ax=ax, label='Predicted Class')
            save_fig(fig, save_dir, "stroke_umap")
            writer.add_figure(f'{phase}/stroke_umap', fig, epoch)
        except Exception as e:
            print(f"UMAP failed (stroke features): {e}")
