"""Training, validation, and prediction helpers for the lab.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import copy
import csv
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, activation_stats, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # filled from the validation sweep in lab.ipynb
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    """
    cm = np.asarray(cm, dtype=np.float64)
    if cm.shape != (7, 7):
        raise ValueError(f"confusion matrix must have shape (7, 7), got {cm.shape}")
    true_positive = np.diag(cm)
    false_positive = cm.sum(axis=0) - true_positive
    false_negative = cm.sum(axis=1) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    class_f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator != 0,
    )
    return float(class_f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits.

    Các bước: model.eval(); duyệt X theo từng lô (không cần xáo); gom argmax(dim=1); torch.cat.
    """
    model.eval()
    predictions = []
    for start in range(0, len(X), batch_size):
        logits = model(X[start : start + batch_size])
        predictions.append(logits.argmax(dim=1))
    if not predictions:
        return torch.empty(0, dtype=torch.int64, device=X.device)
    return torch.cat(predictions).to(dtype=torch.int64)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad.

    Các bước:
      1. model.eval()
      2. tính logits theo từng lô; cộng dồn tổng loss (reduction="sum") rồi chia N cuối cùng
      3. pred = argmax; acc = (pred == y).mean()
      4. dựng ma trận nhầm lẫn 7x7 -> macro_f1_from_confusion
    Dùng hàm này cho: train loss (trên toàn bộ hoặc một tập con CỐ ĐỊNH của train), val, và eval cuối cùng.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")
    if len(X) == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    model.eval()
    total_loss = 0.0
    confusion = torch.zeros((7, 7), dtype=torch.int64, device=y.device)
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = X[start : start + batch_size]
            yb = y[start : start + batch_size]
            logits = model(xb)
            batch_loss = compute_loss(logits, yb, loss_name)
            total_loss += float(batch_loss.item()) * len(yb)
            pred = logits.argmax(dim=1)
            bins = torch.bincount(yb * 7 + pred, minlength=49).reshape(7, 7)
            confusion += bins
    cm = confusion.cpu().numpy()
    accuracy = float(np.trace(cm) / cm.sum())
    return {
        "loss": total_loss / len(X),
        "acc": accuracy,
        "macro_f1": macro_f1_from_confusion(cm),
    }


def compute_loss(logits, y, loss_name: str):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : MSE giữa logit và one-hot của y (ghi rõ bạn lấy trung bình thế nào).
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    if loss_name == "mse":
        one_hot = F.one_hot(y, num_classes=logits.shape[1]).to(dtype=logits.dtype)
        return F.mse_loss(logits, one_hot)
    raise ValueError("loss_name must be 'ce' or 'mse'")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}
    (tên khoá của summary trùng tên cột trong experiments.xlsx)

    Các bước:
      0. set_seed(cfg["seed"]); tạo model = MLP(...), assert count_params(model) == EXPECTED_PARAMS[hidden]
         chuyển model lên device; tạo optimizer = build_optimizer(...)
         nếu precision == "fp16": scaler = torch.amp.GradScaler(...)
      1. step0_loss = evaluate(model, X_val, y_val)["loss"]   # TRƯỚC bước cập nhật đầu tiên; kỳ vọng ≈ ln 7
      2. for epoch in 1..epochs:
           model.train()
           for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], generator):
               with torch.autocast(...)  nếu precision != "fp32":   # chỉ bọc forward + loss
                   logits = model(xb); loss = compute_loss(logits, yb, cfg["loss"])
               optimizer.zero_grad(set_to_none=True)
               backward (qua scaler nếu fp16)
               nếu fp16 và có clip: scaler.unscale_(optimizer)  TRƯỚC khi clip
               gn = clip_gradients(model.parameters(), cfg["clip_norm"])   # chuẩn TRƯỚC khi cắt; ghi lại
               bước cập nhật (scaler.step(optimizer); scaler.update() nếu fp16, ngược lại optimizer.step())
               nếu loss là NaN/inf: đặt diverged=True và dừng sớm, ĐỪNG để notebook treo
           cuối epoch (dùng evaluate, chế độ eval):
               train_loss trên toàn bộ train (hoặc 1 tập con CỐ ĐỊNH ~50 000 mẫu), val_loss/val_acc/val_macro_f1
               grad_norm trung bình của epoch; thời gian epoch (torch.cuda.synchronize() nếu dùng GPU)
               nếu val_loss tốt nhất từ trước tới giờ: lưu best_state (bản sao state_dict) và best_epoch
      3. tổng hợp summary tại best_epoch (val_acc, val_macro_f1 lấy ở best_epoch); peak_mem_MB nếu có GPU
    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    if cfg["lr"] is None or cfg["lr"] <= 0:
        raise ValueError("cfg['lr'] must be selected using validation and be positive")
    if cfg["epochs"] <= 0 or cfg["batch"] <= 0:
        raise ValueError("epochs and batch must be positive")
    if cfg["precision"] not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be 'fp32', 'fp16', or 'bf16'")

    set_seed(int(cfg["seed"]))
    device = data["X_tr"].device
    hidden = tuple(cfg["hidden"])
    if hidden not in EXPECTED_PARAMS:
        raise ValueError(f"No required parameter count is registered for hidden={hidden}")
    model = MLP(hidden=hidden, dropout=float(cfg["dropout"]), init=cfg["init"]).to(device)
    assert count_params(model) == EXPECTED_PARAMS[hidden]
    optimizer = build_optimizer(
        cfg["optimizer"], model.parameters(), lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]), momentum=float(cfg["momentum"]),
    )
    precision = cfg["precision"]
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training requires a CUDA device")
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("This CUDA device does not support bf16")
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    if precision == "fp16":
        try:
            scaler = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):  # Compatibility with early PyTorch 2.x releases.
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    def autocast_context():
        if amp_dtype is None:
            return nullcontext()
        return torch.autocast(device_type=device.type, dtype=amp_dtype)

    step0 = evaluate(model, data["X_val"], data["y_val"], cfg["loss"])
    step0_loss = step0["loss"]
    step0_activation_stds = activation_stats(model, data["X_val"][:1024])
    history = {
        "epoch": [], "train_loss": [], "val_loss": [], "val_acc": [],
        "val_macro_f1": [], "grad_norm": [], "grad_norm_p95": [],
        "grad_norm_max": [], "clip_fraction": [], "epoch_time_s": [],
    }
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    best_val_loss = float("inf")
    best_epoch = 0
    best_metrics = step0
    diverged = False
    epoch_times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    generator = torch.Generator(device=device)
    generator.manual_seed(int(cfg["seed"]))
    train_count = min(len(data["X_tr"]), 50_000)
    train_X_for_metric = data["X_tr"][:train_count]
    train_y_for_metric = data["y_tr"][:train_count]
    initial_train_loss = evaluate(model, train_X_for_metric, train_y_for_metric, cfg["loss"])["loss"]

    for epoch in range(1, int(cfg["epochs"]) + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()
        model.train()
        batch_grad_norms = []
        clipped_batches = 0
        epoch_failed = False
        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], int(cfg["batch"]), generator=generator):
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])
            if not torch.isfinite(loss):
                diverged = epoch_failed = True
                break
            if scaler is not None:
                scaler.scale(loss).backward()
                # Unscale before measuring grad_norm, even when clipping is disabled.
                scaler.unscale_(optimizer)
                grad_norm = clip_gradients(model.parameters(), cfg["clip_norm"])
                if not np.isfinite(grad_norm):
                    diverged = epoch_failed = True
                    break
                if cfg["clip_norm"] is not None and grad_norm > cfg["clip_norm"]:
                    clipped_batches += 1
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = clip_gradients(model.parameters(), cfg["clip_norm"])
                if not np.isfinite(grad_norm):
                    diverged = epoch_failed = True
                    break
                if cfg["clip_norm"] is not None and grad_norm > cfg["clip_norm"]:
                    clipped_batches += 1
                optimizer.step()
            batch_grad_norms.append(grad_norm)

        if epoch_failed:
            break
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - epoch_start
        train_metrics = evaluate(model, train_X_for_metric, train_y_for_metric, cfg["loss"])
        val_metrics = evaluate(model, data["X_val"], data["y_val"], cfg["loss"])
        if not np.isfinite(train_metrics["loss"]) or not np.isfinite(val_metrics["loss"]):
            diverged = True
            break

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["grad_norm"].append(float(np.mean(batch_grad_norms)) if batch_grad_norms else 0.0)
        history["grad_norm_p95"].append(float(np.percentile(batch_grad_norms, 95)) if batch_grad_norms else 0.0)
        history["grad_norm_max"].append(float(np.max(batch_grad_norms)) if batch_grad_norms else 0.0)
        history["clip_fraction"].append(clipped_batches / len(batch_grad_norms) if batch_grad_norms else 0.0)
        history["epoch_time_s"].append(elapsed)
        epoch_times.append(elapsed)
        print(
            f"{cfg['exp_id']} epoch {epoch}/{cfg['epochs']}: "
            f"train_loss={train_metrics['loss']:.4f}, val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics['acc']:.4f}, val_macro_f1={val_metrics['macro_f1']:.4f}, "
            f"grad_norm={history['grad_norm'][-1]:.4g}, clip_fraction={history['clip_fraction'][-1]:.3f}, "
            f"time={elapsed:.1f}s"
        )
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_metrics = val_metrics
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    if best_epoch == 0:
        best_val_loss = step0_loss
    model.load_state_dict(best_state)
    final_train_loss = history["train_loss"][-1] if history["train_loss"] else initial_train_loss
    final_val_loss = history["val_loss"][-1] if history["val_loss"] else step0_loss
    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
    )
    result = {
        "cfg": cfg,
        "history": history,
        "summary": {
            "step0_loss": step0_loss,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "val_acc": best_metrics["acc"],
            "val_macro_f1": best_metrics["macro_f1"],
            "time_per_epoch_s": float(np.mean(epoch_times)) if epoch_times else 0.0,
            "peak_mem_MB": float(peak_mem_mb),
            "diverged": bool(diverged),
            "activation_std_after_relu_step0": step0_activation_stds,
        },
        "best_state": best_state,
    }
    return result


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    row_ids = np.asarray(row_id, dtype=np.int64)
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    predictions = np.asarray(preds, dtype=np.int64)
    if row_ids.ndim != 1 or predictions.ndim != 1 or len(row_ids) != len(predictions):
        raise ValueError("row_id and preds must be 1-D arrays of equal length")
    if len(np.unique(row_ids)) != len(row_ids):
        raise ValueError("row_id values must be unique")
    if len(predictions) and (predictions.min() < 0 or predictions.max() > 6):
        raise ValueError("predictions must be integer labels in 0..6")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("row_id", "pred"))
        writer.writerows(zip(row_ids.tolist(), predictions.tolist()))


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions.

    Các bước:
      1. model = MLP(...); model.load_state_dict(result["best_state"]); lên device
      2. preds = predict(model, data["X_eval"])  # fp32, eval mode
      3. write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
      4. chạy `python scripts/evaluate.py --pred <pred_path>` và ghi kết quả vào bảng/báo cáo
    """
    cfg = {**DEFAULT_CFG, **cfg}
    device = data["X_eval"].device
    hidden = tuple(cfg["hidden"])
    model = MLP(hidden=hidden, dropout=float(cfg["dropout"]), init=cfg["init"]).to(device)
    model.load_state_dict(result["best_state"])
    predictions = predict(model, data["X_eval"])
    write_predictions(data["eval_row_id"], predictions, pred_path)
