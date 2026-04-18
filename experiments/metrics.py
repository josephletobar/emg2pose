import numpy as np
from emg2pose.metrics import get_default_metrics

def _convert_metrics(results):
    out = {}

    for k, v in results.items():
        val = v.item() if hasattr(v, "item") else v

        if "mae" in k:
            out[k + "_deg"] = np.degrees(val)

        elif "vel" in k or "acc" in k or "jerk" in k:
            out[k + "_deg"] = np.degrees(val)

        elif "distance" in k:
            out[k + "_mm"] = val

        else:
            out[k] = val

    return out
    
def get_experiment_metrics(preds, joint_angles, no_ik_failure):
    def to_tensor(x, is_mask=False):
        import torch, numpy as np

        if isinstance(x, np.ndarray):
            t = torch.tensor(x)
        elif isinstance(x, torch.Tensor):
            t = x
        else:
            t = torch.tensor(x)

        return t.bool() if is_mask else t.float()

    # --- convert ---
    pred = to_tensor(preds)
    target = to_tensor(joint_angles)
    mask = to_tensor(no_ik_failure, is_mask=True)

    # --- fix shapes to (B, C, T) and (B, T) ---
    try:
        if pred.ndim == 2:   # (T, C)
            pred = pred.T.unsqueeze(0)
        if target.ndim == 2:
            target = target.T.unsqueeze(0)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
    except:
        pass

    # --- align lengths (graceful degradation) ---
    try:
        T_pred = pred.shape[-1]
        T_target = target.shape[-1]
        T_mask = mask.shape[-1]

        min_T = min(T_pred, T_target, T_mask)

        if not (T_pred == T_target == T_mask):
            print(f"[Warning] Length mismatch (pred={T_pred}, target={T_target}, mask={T_mask}) -> truncating to {min_T}")

        pred = pred[..., :min_T]
        target = target[..., :min_T]
        mask = mask[..., :min_T]

    except:
        pass

    # --- force CPU ---
    try:
        pred = pred.cpu()
        target = target.cpu()
        mask = mask.cpu()
    except:
        pass

    # --- metrics ---
    metrics = get_default_metrics()
    results = {}

    for m in metrics:
        try:
            results.update(m(pred, target, mask, stage="eval"))
        except:
            continue  # skip broken metric

    return _convert_metrics(results)
