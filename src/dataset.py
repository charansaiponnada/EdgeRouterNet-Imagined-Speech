import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import traceback

def slice_window(X, off, TA):
    # X: [trials, channels, time]
    # off: offset in samples
    # TA: Total Analysis window length
    return X[:, :, off:off+TA]

class EEGTransform:
    def __init__(self, noise_std=0.01, channel_dropout_prob=0.1):
        self.noise_std = noise_std
        self.channel_dropout_prob = channel_dropout_prob

    def __call__(self, x):
        # x: [channels, time]
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        
        if self.channel_dropout_prob > 0:
            mask = torch.rand(x.shape[0], 1) > self.channel_dropout_prob
            x = x * mask.float()
        return x

class AllCropsDS(Dataset):
    def __init__(self, X_nct, y, crop_starts, length_map, short_remap, long_remap, transform=None):
        self.X = X_nct
        self.y = y
        self.crop_starts = crop_starts
        self.crop_len = 256
        self.length_map = length_map
        self.short_remap = short_remap
        self.long_remap = long_remap
        self.transform = transform

    def __len__(self):
        return self.X.shape[0] * len(self.crop_starts)

    def __getitem__(self, idx):
        trial_idx = idx // len(self.crop_starts)
        crop_idx = idx % len(self.crop_starts)
        s = self.crop_starts[crop_idx]
        x = self.X[trial_idx, :, s:s+self.crop_len]
        
        x_tensor = torch.from_numpy(x).float()
        if self.transform:
            x_tensor = self.transform(x_tensor)
            
        y_full = int(self.y[trial_idx])
        y_len = self.length_map[y_full]
        
        # routing target
        if y_len == 0: # short
            y_route = self.short_remap[y_full]
        else: # long
            y_route = self.long_remap[y_full]
            
        return x_tensor, torch.tensor(y_full).long(), torch.tensor(y_len).long(), torch.tensor(y_route).long()

def pick_epo_key(keys):
    for k in ["epo_train","epo_val","epo_valid","epo_validation","epo_test","epo"]:
        if k in keys:
            return k
    return None

def unwrap_11_object(v, max_depth=30):
    v = np.array(v)
    for _ in range(max_depth):
        if v.dtype == object:
            v = np.array(v.item())
        elif isinstance(v, np.ndarray) and v.shape == (1,1):
            v = np.array(v[0,0])
        else:
            break
    return v

def load_one_mat_full(fp, L_need, num_classes=5, allow_no_label=False):
    try:
        from scipy.io import loadmat
        d = loadmat(fp)
        epo_key = pick_epo_key(d.keys())
        if epo_key is None:
            raise KeyError(f"No epo_* key in {fp}. Keys: {list(d.keys())}")

        epo = d[epo_key][0,0]
        x = unwrap_11_object(epo["x"]).astype(np.float32)
        t = unwrap_11_object(epo["t"]).squeeze()

        y0 = None
        if ("y" in epo.dtype.names):
            y = unwrap_11_object(epo["y"])
            if y is not None and y.size > 0:
                y0 = np.argmax(y, axis=0).astype(np.int64)

        if (y0 is None) and (not allow_no_label):
            raise KeyError(f"Labels missing in {fp} but allow_no_label=False")

        start_idx = np.where(t >= 0)[0]
        start0 = int(start_idx[0]) if start_idx.size > 0 else 0

        seg = x[start0:start0+L_need]
        if seg.shape[0] < L_need:
            pad = np.zeros((L_need - seg.shape[0], seg.shape[1], seg.shape[2]), dtype=seg.dtype)
            seg = np.concatenate([seg, pad], axis=0)

        X_ncl = np.transpose(seg, (2,1,0)).astype(np.float32)
        return X_ncl, y0

    except NotImplementedError:
        import h5py

        def _read_h5_obj(f, obj):
            if isinstance(obj, h5py.Dataset):
                data = obj[()]
                if data.dtype == object or (hasattr(data, 'dtype') and getattr(data.dtype, 'kind', None) == 'O'):
                    out = np.empty(data.shape, dtype=object)
                    it = np.nditer(data, flags=['multi_index', 'refs_ok'])
                    for r in it:
                        ref = r.item()
                        out[it.multi_index] = _read_h5_obj(f, f[ref])
                    return out
                return data
            if isinstance(obj, h5py.Group):
                return {k: _read_h5_obj(f, obj[k]) for k in obj.keys()}
            return obj

        with h5py.File(fp, "r") as f:
            epo_key = pick_epo_key(f.keys())
            if epo_key is None:
                raise KeyError(f"No epo_* key in v7.3 file {fp}. Keys: {list(f.keys())}")

            epo = _read_h5_obj(f, f[epo_key])
            x = np.array(epo["x"]).astype(np.float32)
            t = np.array(epo["t"]).squeeze()

            y0 = None
            if "y" in epo:
                y = np.array(epo["y"])
                if y.size > 0:
                    if y.shape[0] == num_classes:
                        y0 = np.argmax(y, axis=0).astype(np.int64)
                    elif y.shape[-1] == num_classes:
                        y0 = np.argmax(y, axis=-1).astype(np.int64)

            if (y0 is None) and (not allow_no_label):
                raise KeyError(f"Labels missing in {fp} but allow_no_label=False")

            if x.shape[1] != 64 and x.shape[0] == 64:
                x = np.transpose(x, (1,0,2))
            elif x.shape[1] != 64 and x.shape[2] == 64:
                x = np.transpose(x, (0,2,1))

            start_idx = np.where(t >= 0)[0]
            start0 = int(start_idx[0]) if start_idx.size > 0 else 0

            seg = x[start0:start0+L_need]
            if seg.shape[0] < L_need:
                pad = np.zeros((L_need - seg.shape[0], seg.shape[1], seg.shape[2]), dtype=seg.dtype)
                seg = np.concatenate([seg, pad], axis=0)

            X_ncl = np.transpose(seg, (2,1,0)).astype(np.float32)
            return X_ncl, y0

def load_split_full(path, name, L_need, num_classes=5, allow_no_label=False):
    files = sorted(glob.glob(os.path.join(path, "*.mat")))
    if len(files) == 0:
        print(f"WARNING: No .mat files found in {path}")
        return None, None, []

    Xs, ys = [], []
    for fp in files:
        try:
            X, y = load_one_mat_full(fp, L_need, num_classes, allow_no_label)
            Xs.append(X)
            if y is not None:
                ys.append(y)
        except Exception as e:
            print(f"[{name}] BAD FILE: {fp}\n{repr(e)}")

    if len(Xs) == 0:
        return None, None, []

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0) if len(ys) else None
    return X, y, files
