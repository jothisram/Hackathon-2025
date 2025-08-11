# detect_ring_defects_threshold.py
import cv2
import numpy as np
import glob, os, math
from scipy.signal import savgol_filter

def mad(a):
    a = np.asarray(a)
    return np.median(np.abs(a - np.median(a))) if a.size else 0.0

def radial_profiles(mask, center, n_angles=720, max_r=None):
    h, w = mask.shape
    if max_r is None:
        max_r = int(math.hypot(w, h)) // 2 + 10
    cx, cy = center
    thetas = np.linspace(0, 2 * math.pi, n_angles, endpoint=False)
    r_in = np.zeros(n_angles, dtype=float)
    r_out = np.zeros(n_angles, dtype=float)
    for i, th in enumerate(thetas):
        r_first = None
        r_last = None
        for r in range(0, max_r):
            x = int(round(cx + r * math.cos(th)))
            y = int(round(cy + r * math.sin(th)))
            if x < 0 or y < 0 or x >= w or y >= h:
                break
            val = 1 if mask[y, x] > 0 else 0
            if val == 1 and r_first is None:
                r_first = r
            if val == 1:
                r_last = r
        r_in[i] = np.nan if r_first is None else r_first
        r_out[i] = np.nan if r_last is None else r_last
    return thetas, r_in, r_out

def merge_runs(bool_arr, min_len=4):
    n = len(bool_arr)
    runs = []
    extended = np.concatenate([bool_arr, bool_arr])
    i = 0
    while i < len(extended):
        if extended[i]:
            start = i
            while i < len(extended) and extended[i]:
                i += 1
            end = i
            if end - start >= min_len:
                s = start % n
                e = (end - 1) % n
                runs.append((s, e))
        else:
            i += 1
    cleaned = []
    for (s, e) in runs:
        if s <= e:
            length = e - s + 1
            cleaned.append((s, e, length))
        else:
            length = (n - s) + (e + 1)
            cleaned.append((s, e, length))
    return cleaned

def detect_defects_from_profiles(thetas, r_in, r_out, k=4, min_deg_span=1.0):
    n = len(thetas)

    def fillnan(a):
        a = np.array(a, dtype=float)
        mask = np.isnan(a)
        if mask.all():
            return a
        x = np.arange(n)
        a[mask] = np.interp(x[mask], x[~mask], a[~mask])
        return a

    r_in_f = fillnan(r_in)
    r_out_f = fillnan(r_out)

    win = 21 if n >= 21 else (n // 2 * 2 + 1)
    if win < 5:
        win = 5
    try:
        r_in_hat = savgol_filter(r_in_f, win, 2)
        r_out_hat = savgol_filter(r_out_f, win, 2)
    except Exception:
        r_in_hat = r_in_f
        r_out_hat = r_out_f

    e_out = r_out_f - r_out_hat
    e_in = r_in_hat - r_in_f  # positive if inward defect (cut)

    tau_out = k * mad(e_out[~np.isnan(e_out)])
    tau_in = k * mad(e_in[~np.isnan(e_in)])

    mask_out = (np.abs(e_out) > tau_out) & (~np.isnan(e_out))
    mask_in = (np.abs(e_in) > tau_in) & (~np.isnan(e_in))

    samples_per_deg = n / 360.0
    min_len = max(2, int(np.ceil(min_deg_span * samples_per_deg)))
    runs_out = merge_runs(mask_out, min_len=min_len)
    runs_in = merge_runs(mask_in, min_len=min_len)

    defects = []
    # Outer edge defects
    for (s, e, l) in runs_out:
        idxs = [(s + i) % n for i in range(l)]
        vals = e_out[idxs]
        idx = idxs[int(np.argmax(np.abs(vals)))]
        defect_type = "flash" if e_out[idx] > 0 else "cut"
        defects.append({'edge': 'outer', 'peak_idx': idx, 'depth': float(vals.max()), 'type': defect_type})
    # Inner edge defects
    for (s, e, l) in runs_in:
        idxs = [(s + i) % n for i in range(l)]
        vals = e_in[idxs]
        idx = idxs[int(np.argmax(np.abs(vals)))]
        defect_type = "cut" if e_in[idx] > 0 else "flash"
        defects.append({'edge': 'inner', 'peak_idx': idx, 'depth': float(vals.max()), 'type': defect_type})

    return defects, r_in_hat, r_out_hat

def annotate_and_save(img, center, thetas, r_in, r_out, defects, outpath):
    out = img.copy()
    for d in defects:
        idx = d['peak_idx']
        theta = thetas[idx]
        r = r_out[idx] if d['edge'] == 'outer' else r_in[idx]
        if np.isnan(r):
            continue
        x = int(round(center[0] + r * math.cos(theta)))
        y = int(round(center[1] + r * math.sin(theta)))

        if d['type'] == 'flash':
            color = (0, 255, 0)  # Green for flash
        else:
            color = (0, 0, 255)  # Red for cut

        cv2.circle(out, (x, y), 6, color, -1)
        cv2.putText(out, d['type'], (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    cv2.imwrite(outpath, out)
    return out


def process_file(fpath):
    img = cv2.imread(fpath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bin_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    mask = bin_inv.copy()

    M = cv2.moments(mask)
    if M['m00'] == 0:
        cx = mask.shape[1] // 2
        cy = mask.shape[0] // 2
    else:
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']

    thetas, r_in, r_out = radial_profiles(mask, (cx, cy), n_angles=720)
    defects, r_in_hat, r_out_hat = detect_defects_from_profiles(thetas, r_in, r_out, k=4, min_deg_span=1.0)

    outname = "output_" + os.path.basename(fpath)
    annotate_and_save(img, (cx, cy), thetas, r_in, r_out, defects, outname)

    return defects, outname

if __name__ == "__main__":
    pngs = sorted(glob.glob("*.png"))
    if not pngs:
        print("No PNG images found in current folder.")
    for p in pngs:
        defects, outimg = process_file(p)
        print("==", p, "==")
        if not defects:
            print("  No defects detected")
        else:
            for d in defects:
                idx = d['peak_idx']
                deg = (idx * 360.0 / 720) % 360
                typ = "flash" if d['edge'] == 'outer' else "cut"
                print(f"  - {typ} at {deg:.1f}° depth {d['depth']:.1f}")
        print("Annotated saved to:", outimg)