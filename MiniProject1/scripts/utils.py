import torch
import os
import json
import time


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def _history_to_serializable(history):
    serial = {}
    for k, v in history.items():
        serial[k] = [float(x) for x in v]
    return serial

def save_checkpoint(save_dir, model, optimizer, epoch, history, prefix="model"):
    ensure_dir(save_dir)

    ckpt_path = os.path.join(save_dir, f"{prefix}_epoch{epoch}.pth")
    history_path = os.path.join(save_dir, f"{prefix}_history_epoch{epoch}.json")

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
    }
    torch.save(checkpoint, ckpt_path)

    serial_history = _history_to_serializable(history)
    with open(history_path, "w") as f:
        json.dump(serial_history, f, indent=2)

    latest_ckpt = os.path.join(save_dir, f"{prefix}_latest.pth")
    latest_history = os.path.join(save_dir, f"{prefix}_history_latest.json")
    torch.save(checkpoint, latest_ckpt)
    with open(latest_history, "w") as f:
        json.dump(serial_history, f, indent=2)

    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Saved history:    {history_path}")

def load_checkpoint(ckpt_path, model, optimizer=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch = ckpt.get("epoch", None)
    return epoch

def load_history(history_path):
    with open(history_path, "r") as f:
        hist = json.load(f)
    for k in list(hist.keys()):
        hist[k] = [float(x) for x in hist[k]]
    return hist

def evaluate_on_device(model, dataloader, device, repeat=1):
    model.to(device)
    model.eval()

    total_time = 0.0
    total_samples = 0

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total

    for r in range(repeat):
        start_time = time.time()

        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(device)
                _ = model(images)
                total_samples += images.size(0)

        end_time = time.time()
        total_time += (end_time - start_time)

    avg_time_per_run = total_time / repeat
    avg_time_per_sample = total_time / (total_samples if total_samples > 0 else 1)

    print(f"Device: {device}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Repeats: {repeat}")
    print(f"  Total time: {total_time:.3f} s")
    print(f"  Avg time per run: {avg_time_per_run:.3f} s")
    print(f"  Avg time per sample: {avg_time_per_sample:.6f} s")
    print(f"  Total sample: {total_samples}")

    return accuracy, avg_time_per_sample