"""Training pipeline with MLflow experiment tracking."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.optim import Adam

from src import config
from src.data.dataset import build_dataloaders
from src.data.download import ensure_dataset
from src.data.preprocess import load_splits, prepare_dataset
from src.models import build_model


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _runtime_params(device: torch.device) -> dict:
    params = {
        "device": str(device),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        params["gpu_name"] = torch.cuda.get_device_name(0)
    return params


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * images.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    n = max(len(all_labels), 1)
    metrics = {
        "loss": total_loss / n,
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
    }
    return metrics, all_labels, all_preds


def _plot_confusion_matrix(y_true, y_pred, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=list(config.CLASS_NAMES),
        yticklabels=list(config.CLASS_NAMES),
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_loss_curves(history: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train_loss"], label="train")
    ax.plot(history["val_loss"], label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_checkpoint(model, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "meta": meta,
        "class_names": list(config.CLASS_NAMES),
        "image_size": config.IMAGE_SIZE,
    }
    torch.save(payload, path)


def train(
    epochs: int = config.DEFAULT_EPOCHS,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
    lr: float = config.DEFAULT_LR,
    smoke: bool = False,
) -> dict:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    config.MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

    if smoke:
        ensure_dataset(synthetic=True, n_per_class=config.SMOKE_MAX_PER_CLASS)
        prepare_dataset(max_per_class=config.SMOKE_MAX_PER_CLASS)
        epochs = config.SMOKE_EPOCHS
        batch_size = config.SMOKE_BATCH_SIZE
    elif not config.SPLITS_PATH.exists():
        prepare_dataset()

    splits = load_splits()
    loaders = build_dataloaders(splits, batch_size=batch_size)
    device = _device()
    model = build_model(num_classes=config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    mlflow.set_tracking_uri(config.MLRUNS_DIR.as_uri())
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    mlflow.enable_system_metrics_logging()
    try:
        mlflow.set_system_metrics_sampling_interval(1)
        mlflow.set_system_metrics_samples_before_logging(1)
    except Exception:
        pass

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val_acc = -1.0
    best_state = None

    with mlflow.start_run(
        run_name="simple_cnn_smoke" if smoke else "simple_cnn",
        log_system_metrics=True,
    ):
        mlflow.log_params(
            {
                "model": "SimpleCNN",
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "image_size": config.IMAGE_SIZE,
                "smoke": smoke,
                "optimizer": "Adam",
                "train_size": len(splits["train"]),
                "val_size": len(splits["val"]),
                "test_size": len(splits["test"]),
                **_runtime_params(device),
            }
        )

        for epoch in range(1, epochs + 1):
            train_metrics, _, _ = _run_epoch(
                model, loaders["train"], criterion, optimizer, device, train=True
            )
            val_metrics, _, _ = _run_epoch(
                model, loaders["val"], criterion, optimizer, device, train=False
            )
            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1": val_metrics["f1"],
                },
                step=epoch,
            )
            print(
                f"Epoch {epoch}/{epochs} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )
            if val_metrics["accuracy"] >= best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        test_metrics, y_true, y_pred = _run_epoch(
            model, loaders["test"], criterion, optimizer, device, train=False
        )
        mlflow.log_metrics(
            {
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
            }
        )

        cm_path = config.ARTIFACTS_DIR / "confusion_matrix.png"
        loss_path = config.ARTIFACTS_DIR / "loss_curves.png"
        _plot_confusion_matrix(y_true, y_pred, cm_path)
        _plot_loss_curves(history, loss_path)
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(loss_path))

        meta = {
            "test_metrics": {k: v for k, v in test_metrics.items()},
            "best_val_accuracy": best_val_acc,
            "smoke": smoke,
            "image_size": config.IMAGE_SIZE,
        }
        save_checkpoint(model, config.CHAMPION_MODEL_PATH, meta)
        mlflow.log_artifact(str(config.CHAMPION_MODEL_PATH))

        metrics_path = config.ARTIFACTS_DIR / "metrics.json"
        metrics_path.write_text(json.dumps(meta, indent=2))
        mlflow.log_artifact(str(metrics_path))

        print(f"Champion model saved to {config.CHAMPION_MODEL_PATH}")
        print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Cats vs Dogs CNN")
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LR)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast synthetic-data run for CI",
    )
    args = parser.parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, smoke=args.smoke)


if __name__ == "__main__":
    main()
