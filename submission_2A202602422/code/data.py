"""Part 0 data loading, validation splitting, and standardization.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    train_path = f"{processed_dir}/train.npz"
    eval_path = f"{processed_dir}/eval.npz"
    with np.load(train_path) as train, np.load(eval_path) as evaluation:
        X_train = train["X"]
        y_train = train["y"]
        X_eval = evaluation["X"]
        y_eval = evaluation["y"]
        eval_row_id = evaluation["row_id"]

    assert X_train.ndim == X_eval.ndim == 2 and X_train.shape[1] == X_eval.shape[1] == 54
    assert X_train.dtype == X_eval.dtype == np.float32
    assert y_train.shape == (len(X_train),) and y_eval.shape == (len(X_eval),)
    assert y_train.dtype == y_eval.dtype == np.int64
    assert eval_row_id.shape == (len(X_eval),) and eval_row_id.dtype == np.int64
    assert np.isfinite(X_train).all() and np.isfinite(X_eval).all()
    assert y_train.min() >= 0 and y_train.max() <= 6
    assert y_eval.min() >= 0 and y_eval.max() <= 6
    return X_train, y_train, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, random_state=seed, stratify=y
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    """
    X_tr = np.asarray(X_tr, dtype=np.float32)
    if X_tr.ndim != 2 or X_tr.shape[1] < N_NUMERIC:
        raise ValueError(f"X_tr must have at least {N_NUMERIC} feature columns")
    mean = X_tr[:, :N_NUMERIC].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X_tr[:, :N_NUMERIC].std(axis=0, dtype=np.float64).astype(np.float32)
    # A constant feature needs no rescaling; using 1 avoids division by zero.
    std[std == 0] = 1.0
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] < N_NUMERIC:
        raise ValueError(f"X must have at least {N_NUMERIC} feature columns")
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != (N_NUMERIC,) or std.shape != (N_NUMERIC,):
        raise ValueError(f"mean and std must both have shape ({N_NUMERIC},)")
    if np.any(std <= 0):
        raise ValueError("std values must be positive; use fit_standardizer to handle constants")
    transformed = X.copy()
    transformed[:, :N_NUMERIC] = (transformed[:, :N_NUMERIC] - mean) / std
    return transformed


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(
        X_train_full, y_train_full, val_fraction=val_fraction, seed=seed
    )
    mean, std = fit_standardizer(X_tr)
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    device = torch.device(device)
    data = {
        "X_tr": torch.as_tensor(X_tr, dtype=torch.float32, device=device),
        "y_tr": torch.as_tensor(y_tr, dtype=torch.int64, device=device),
        "X_val": torch.as_tensor(X_val, dtype=torch.float32, device=device),
        "y_val": torch.as_tensor(y_val, dtype=torch.int64, device=device),
        "X_eval": torch.as_tensor(X_eval, dtype=torch.float32, device=device),
        "y_eval": torch.as_tensor(y_eval, dtype=torch.int64, device=device),
        "eval_row_id": eval_row_id,
        "standardizer_mean": mean,
        "standardizer_std": std,
    }

    majority_label = int(np.bincount(y_tr, minlength=7).argmax())
    val_majority_accuracy = float(np.mean(y_val == majority_label))
    print(f"train: {tuple(data['X_tr'].shape)}, val: {tuple(data['X_val'].shape)}, eval: {tuple(data['X_eval'].shape)}")
    print(f"device: {device}; majority class from train: {majority_label}; val majority accuracy: {val_majority_accuracy:.4f}")
    print(f"standardizer train mean (10 numeric features): {data['X_tr'][:, :N_NUMERIC].mean(dim=0).detach().cpu().numpy().round(4)}")
    print(f"standardizer train std  (10 numeric features): {data['X_tr'][:, :N_NUMERIC].std(dim=0, unbiased=False).detach().cpu().numpy().round(4)}")
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; hãy quyết định bạn xử lý thế nào và ghi lại.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples")
    n_samples = len(X)
    if shuffle:
        perm = torch.randperm(n_samples, generator=generator, device=X.device)
    else:
        perm = torch.arange(n_samples, device=X.device)
    for start in range(0, n_samples, batch_size):
        indices = perm[start : start + batch_size]
        yield X[indices], y[indices]
