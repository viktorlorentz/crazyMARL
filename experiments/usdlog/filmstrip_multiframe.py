#!/usr/bin/env python3
# filmstrip_trail_vertical_panels.py
import argparse
import cv2
import numpy as np

# ---------------- I/O & Prep ----------------

def iter_frames(path, rotate="cw"):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    rot_map = {"none": None, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "cw": cv2.ROTATE_90_CLOCKWISE}
    rot_mode = rot_map.get(rotate, cv2.ROTATE_90_CLOCKWISE)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if rot_mode is not None:
            f = cv2.rotate(f, rot_mode)
        frames.append(f)
    cap.release()
    if not frames:
        raise SystemExit("No frames decoded.")
    return frames

def sample_indices(total, num_panels):
    """Evenly sample 'num_panels' indices across [0..total-1]."""
    if num_panels <= 1:
        return [total - 1]
    xs = np.linspace(0, total - 1, num_panels)
    idxs = [int(round(x)) for x in xs]
    # ensure strictly increasing within bounds
    idxs = np.clip(np.maximum.accumulate(idxs), 0, total - 1).tolist()
    return idxs

def build_background(frames, bg_samples=60):
    take = frames[:max(1, min(bg_samples, len(frames)))]
    return np.median(np.stack(take, axis=0), axis=0).astype(np.uint8)

def darken(img_u8, amount):
    """Darken toward black by 'amount' in [0..1]. 0=no change, 1=black."""
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0:
        return img_u8
    out = (img_u8.astype(np.float32) * (1.0 - amount)).clip(0, 255).astype(np.uint8)
    return out

# ---------------- Motion & Masks ----------------

def motion_mask(frame, bg_ref, blur_sigma=1.2, thresh=22, open_ksize=3, dilate_iter=1, min_area=80):
    diff = cv2.absdiff(frame, bg_ref)
    g = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    if blur_sigma and blur_sigma > 0:
        g = cv2.GaussianBlur(g, (0, 0), blur_sigma)
    _, m = cv2.threshold(g, thresh, 255, cv2.THRESH_BINARY)
    if open_ksize and open_ksize > 1:
        k = np.ones((open_ksize, open_ksize), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    if dilate_iter and dilate_iter > 0:
        m = cv2.dilate(m, None, iterations=dilate_iter)
    if min_area and min_area > 0:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        keep = np.zeros_like(m)
        for c in cnts:
            if cv2.contourArea(c) >= min_area:
                cv2.drawContours(keep, [c], -1, 255, thickness=cv2.FILLED)
        m = keep
    return m  # uint8(0/255)

def latest_mask_thinwire(frame, bg_ref, thin_thresh=8,
                         blur_sigma=1.2, open_ksize_latest=0, dilate_iter_latest=1,
                         edge_boost=True, canny1=40, canny2=120):
    """
    Sensitive mask for most recent frame:
    - Low-threshold diff vs ORIGINAL bg (so darkening doesn't affect detection)
    - Optional Canny edges union to catch thin wires.
    """
    diff = cv2.absdiff(frame, bg_ref)
    g = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    if blur_sigma and blur_sigma > 0:
        g = cv2.GaussianBlur(g, (0, 0), blur_sigma)
    _, m_low = cv2.threshold(g, thin_thresh, 255, cv2.THRESH_BINARY)
    if edge_boost:
        edges = cv2.Canny(g, canny1, canny2)
        m_thin = cv2.max(m_low, edges)
    else:
        m_thin = m_low
    if open_ksize_latest and open_ksize_latest > 1:
        k = np.ones((open_ksize_latest, open_ksize_latest), np.uint8)
        m_thin = cv2.morphologyEx(m_thin, cv2.MORPH_OPEN, k, iterations=1)
    if dilate_iter_latest and dilate_iter_latest > 0:
        m_thin = cv2.dilate(m_thin, None, iterations=dilate_iter_latest)
    return m_thin

def bbox_from_mask(mask_u8, pad=12, fallback=None, img_shape=None):
    ys, xs = np.where(mask_u8 > 0)
    if ys.size == 0 or xs.size == 0:
        return fallback
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()
    if img_shape is not None:
        H, W = img_shape[:2]
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(W - 1, x2 + pad); y2 = min(H - 1, y2 + pad)
    return (x1, y1, x2, y2)

# ---------------- Cropping ----------------

def fit_bbox_to_aspect(bbox, target_w, target_h, img_w, img_h):
    """Expand bbox to match target aspect, clamp to frame."""
    x1, y1, x2, y2 = bbox
    bw = x2 - x1 + 1
    bh = y2 - y1 + 1
    cx = x1 + bw // 2
    cy = y1 + bh // 2

    ar_tgt = float(target_w) / float(target_h)
    ar_box = float(bw) / float(bh)

    if ar_box > ar_tgt:
        bh = int(round(bw / ar_tgt))
    else:
        bw = int(round(bh * ar_tgt))

    nx1 = cx - bw // 2; ny1 = cy - bh // 2
    nx2 = nx1 + bw - 1; ny2 = ny1 + bh - 1

    # clamp
    if nx1 < 0: nx2 -= nx1; nx1 = 0
    if ny1 < 0: ny2 -= ny1; ny1 = 0
    if nx2 >= img_w: sh = nx2 - img_w + 1; nx1 -= sh; nx2 -= sh
    if ny2 >= img_h: sh = ny2 - img_h + 1; ny1 -= sh; ny2 -= sh

    nx1 = max(0, nx1); ny1 = max(0, ny1)
    nx2 = min(img_w - 1, nx2); ny2 = min(img_h - 1, ny2)
    return (nx1, ny1, nx2, ny2)

def center_box(center_xy, target_w, target_h, img_w, img_h, box_scale=0.8):
    """Crop centered at (cx,cy) with size ~ box_scale * min(img_w,img_h), adjusted to target AR."""
    cx, cy = center_xy
    base = max(1, int(min(img_w, img_h) * np.clip(box_scale, 1e-3, 1.0)))
    bw = base; bh = base
    ar_tgt = float(target_w) / float(target_h)
    if (bw / bh) > ar_tgt:
        bh = int(round(bw / ar_tgt))
    else:
        bw = int(round(bh * ar_tgt))
    x1 = cx - bw // 2; y1 = cy - bh // 2
    x2 = x1 + bw - 1; y2 = y1 + bh - 1
    # clamp
    if x1 < 0: x2 -= x1; x1 = 0
    if y1 < 0: y2 -= y1; y1 = 0
    if x2 >= img_w: sh = x2 - img_w + 1; x1 -= sh; x2 -= sh
    if y2 >= img_h: sh = y2 - img_h + 1; y1 -= sh; y2 -= sh
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img_w - 1, x2); y2 = min(img_h - 1, y2)
    return (x1, y1, x2, y2)

# ---------------- Compositing ----------------

def paste_with_alpha(dst, src, mask_u8, alpha):
    """Blend src → dst where mask==255; alpha=1 => hard overwrite."""
    out = dst.copy()
    mask = mask_u8.astype(bool)
    if not np.any(mask):
        return out
    if alpha >= 1.0:
        out[mask] = src[mask]
        return out
    blended = (alpha * src + (1.0 - alpha) * out).astype(out.dtype)
    out[mask] = blended[mask]
    return out

def gaussian_blur_masked(src, mask_u8, sigma):
    """Blur only masked regions."""
    if sigma <= 0:
        return src
    blurred = cv2.GaussianBlur(src, (0, 0), sigma)
    out = src.copy()
    m = mask_u8.astype(bool)
    out[m] = blurred[m]
    return out

# ---------------- Trail Composition ----------------

def window_indices(s, e, stride):
    """Return indices from s..e inclusive with step=stride, ensuring 'e' is included."""
    if stride <= 1:
        return list(range(s, e + 1))
    ks = list(range(s, e + 1, stride))
    if ks[-1] != e:
        ks.append(e)
    return ks

def make_window_trail_panel(frames, masks_general, bg_img_render, s, e,
                            start_alpha=0.25, end_alpha=1.0, fade_gamma=1.2,
                            trail_blur_sigma=2.5, trail_stride=1,
                            thin_params=None):
    """
    Compose trail for window [s..e] on top of bg_img_render.
    - Use general masks for older frames (optionally blurred).
    - Use thin-wire sensitive mask for the newest frame (union with general).
    - Only include every 'trail_stride' frame in the window; newest always included.
    """
    canvas = bg_img_render.copy()
    if e < s:
        return canvas

    ks = window_indices(s, e, max(1, int(trail_stride)))
    count = len(ks)

    for idx, k in enumerate(ks):
        r = (idx + 1) / count
        a = start_alpha + (end_alpha - start_alpha) * (r ** max(fade_gamma, 1e-6))
        src = frames[k]

        if k != e:
            m_use = masks_general[k]
            if trail_blur_sigma > 0:
                src = gaussian_blur_masked(src, m_use, trail_blur_sigma)
        else:
            # newest frame: thin-wire mask vs ORIGINAL bg, unioned with general mask
            m_thin = latest_mask_thinwire(
                frame=src,
                bg_ref=thin_params["bg_ref_orig"],
                thin_thresh=thin_params["thin_thresh"],
                blur_sigma=thin_params["blur_sigma"],
                open_ksize_latest=thin_params["open_ksize_latest"],
                dilate_iter_latest=thin_params["dilate_iter_latest"],
                edge_boost=thin_params["edge_boost"],
                canny1=thin_params["canny1"],
                canny2=thin_params["canny2"],
            )
            m_use = cv2.max(masks_general[k], m_thin)

        canvas = paste_with_alpha(canvas, src, m_use, a)
    return canvas

# ---------------- Film Strip ----------------

def build_vertical_filmstrip(frames, bg_ref_orig,
                             num_panels=8,
                             # opacity/blur
                             start_alpha=0.25, end_alpha=1.0, fade_gamma=1.2, trail_blur_sigma=2.5,
                             # general mask params
                             thresh=22, blur_sigma=1.2, open_ksize=3, dilate_iter=1, min_area=80,
                             # thin-wire (newest) params
                             thin_thresh=8, open_ksize_latest=0, dilate_iter_latest=1,
                             edge_boost=True, canny1=40, canny2=120,
                             # crop & panel
                             crop_pad=12, manual_center=None, crop_box_scale=0.8,
                             panel_w=None, panel_h=None,
                             # background darkening for rendering
                             darken_bg_amount=0.0,
                             # trail stride
                             trail_stride=1):
    """
    - Evenly sample indices -> idxs
    - Panel i uses frames [prev_idx+1 .. idxs[i]] (first panel: [0..idxs[0]]), with stride.
    - Masks are computed against ORIGINAL background to be robust to darkening.
    - Compositing is done onto a DARKENED background for better contrast.
    """
    H, W = frames[0].shape[:2]
    if panel_w is None: panel_w = W
    if panel_h is None: panel_h = max(1, H // num_panels)

    # Backgrounds
    bg_render = darken(bg_ref_orig, darken_bg_amount)  # used as the canvas
    # Masks vs ORIGINAL bg (fixes the "darken removed trail" issue)
    masks_general = [motion_mask(f, bg_ref_orig, blur_sigma, thresh, open_ksize, dilate_iter, min_area)
                     for f in frames]

    idxs = sample_indices(len(frames), num_panels)
    panels = []
    last_bbox = None

    thin_params = dict(
        thin_thresh=thin_thresh,
        blur_sigma=blur_sigma,
        open_ksize_latest=open_ksize_latest,
        dilate_iter_latest=dilate_iter_latest,
        edge_boost=bool(edge_boost),
        canny1=canny1,
        canny2=canny2,
        bg_ref_orig=bg_ref_orig,  # ALWAYS vs original for sensitivity
    )

    for i, cur in enumerate(idxs):
        prev = (idxs[i - 1] if i > 0 else -1)
        s = prev + 1
        e = cur

        # Compose trail [s..e] with stride on the darkened background
        panel_full = make_window_trail_panel(
            frames=frames,
            masks_general=masks_general,
            bg_img_render=bg_render,
            s=s, e=e,
            start_alpha=start_alpha, end_alpha=end_alpha,
            fade_gamma=fade_gamma, trail_blur_sigma=trail_blur_sigma,
            trail_stride=trail_stride,
            thin_params=thin_params
        )

        # Determine crop
        if manual_center is not None:
            cx, cy = manual_center
            bbox = center_box((cx, cy), panel_w, panel_h, W, H, box_scale=crop_box_scale)
        else:
            # auto: use current mask bbox (vs ORIGINAL bg), then fit to panel AR
            bbox = bbox_from_mask(masks_general[e], pad=crop_pad, fallback=last_bbox, img_shape=(H, W, 3))
            if bbox is None:
                cx, cy = W // 2, (2 * H) // 3
                bw, bh = max(1, W // 3), max(1, H // 3)
                bbox = (max(0, cx - bw // 2), max(0, cy - bh // 2),
                        min(W - 1, cx + bw // 2), min(H - 1, cy + bh // 2))
            bbox = fit_bbox_to_aspect(bbox, panel_w, panel_h, W, H)
            last_bbox = bbox

        x1, y1, x2, y2 = bbox
        crop = panel_full[y1:y2 + 1, x1:x2 + 1]
        panel = cv2.resize(crop, (panel_w, panel_h), interpolation=cv2.INTER_CUBIC)
        panels.append(panel)

    film = np.concatenate(panels, axis=0)
    return film

# ---------------- CLI ----------------

def parse_center(s):
    """Parse 'x,y' into (int,int)."""
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("crop_center must be 'x,y'")
    try:
        x = int(parts[0].strip()); y = int(parts[1].strip())
    except ValueError:
        raise argparse.ArgumentTypeError("crop_center must be 'x,y' with integers")
    return (x, y)

def main():
    ap = argparse.ArgumentParser(
        description="Vertical film strip: N panels; each shows trail since previous panel (with stride); newest opaque (thin-wire safe), older blurred; manual crop & panel size; optional darkened background. Default rotation=cw."
    )
    ap.add_argument("video", help="Input video path")
    ap.add_argument("-o", "--output", default="filmstrip_panels.png", help="Output image path")

    # panels / sampling
    ap.add_argument("--num_panels", type=int, default=8, help="Number of vertical panels (default: 8)")
    ap.add_argument("--rotate", choices=["none", "ccw", "cw"], default="cw", help="Rotate at ingest (default cw)")
    ap.add_argument("--bg_samples", type=int, default=60, help="Frames used for median background")

    # opacity / blur
    ap.add_argument("--start_alpha", type=float, default=0.25, help="Opacity for oldest in window")
    ap.add_argument("--end_alpha", type=float, default=1.0, help="Opacity for newest in window")
    ap.add_argument("--fade_gamma", type=float, default=1.2, help="Opacity ramp (>1 emphasizes newest)")
    ap.add_argument("--trail_blur_sigma", type=float, default=2.5, help="Gaussian sigma to blur older frames in trail")

    # motion mask (general frames)
    ap.add_argument("--thresh", type=int, default=22, help="Intensity threshold for motion")
    ap.add_argument("--blur_sigma", type=float, default=1.2, help="Pre-threshold Gaussian blur sigma")
    ap.add_argument("--open_ksize", type=int, default=3, help="Opening kernel size (0/1 disables)")
    ap.add_argument("--dilate_iter", type=int, default=1, help="Dilate iterations")
    ap.add_argument("--min_area", type=int, default=80, help="Min blob area to keep (px)")
    ap.add_argument("--crop_pad", type=int, default=12, help="Padding around detected bbox (px)")

    # latest-frame thin-wire params
    ap.add_argument("--thin_thresh", type=int, default=8, help="Low threshold for latest-frame thin-wire mask")
    ap.add_argument("--open_ksize_latest", type=int, default=0, help="Opening kernel (latest mask; 0/1 disables)")
    ap.add_argument("--dilate_iter_latest", type=int, default=1, help="Dilate iterations (latest mask)")
    ap.add_argument("--edge_boost", type=int, choices=[0,1], default=1, help="Use Canny edges on latest mask (1=yes)")
    ap.add_argument("--canny1", type=int, default=40, help="Canny lower threshold (latest)")
    ap.add_argument("--canny2", type=int, default=120, help="Canny upper threshold (latest)")

    # manual crop options & panel size
    ap.add_argument("--crop_center", type=parse_center, default=None,
                    help="Override crop center as 'x,y' in pixel coords")
    ap.add_argument("--crop_box_scale", type=float, default=0.8,
                    help="Relative size of crop when --crop_center is set (0..1 of min(frame W,H))")
    ap.add_argument("--panel_w", type=int, default=0, help="Panel width (0 = input frame width)")
    ap.add_argument("--panel_h", type=int, default=0, help="Panel height (0 = H/num_panels)")

    # background darkening
    ap.add_argument("--darken_bg", type=float, default=0.0,
                    help="Darken background by amount in [0..1]; 0=no change, 1=black")

    # trail stride
    ap.add_argument("--trail_stride", type=int, default=1,
                    help="Use every Xth frame in the trail window (newest is always included)")

    args = ap.parse_args()

    frames = iter_frames(args.video, rotate=args.rotate)
    H, W = frames[0].shape[:2]
    bg_ref = build_background(frames, bg_samples=args.bg_samples)

    panel_w = args.panel_w if args.panel_w > 0 else W
    panel_h = args.panel_h if args.panel_h > 0 else max(1, H // args.num_panels)

    film = build_vertical_filmstrip(
        frames=frames, bg_ref_orig=bg_ref,
        num_panels=args.num_panels,
        start_alpha=args.start_alpha, end_alpha=args.end_alpha, fade_gamma=args.fade_gamma,
        trail_blur_sigma=args.trail_blur_sigma,
        thresh=args.thresh, blur_sigma=args.blur_sigma, open_ksize=args.open_ksize,
        dilate_iter=args.dilate_iter, min_area=args.min_area,
        thin_thresh=args.thin_thresh, open_ksize_latest=args.open_ksize_latest,
        dilate_iter_latest=args.dilate_iter_latest,
        edge_boost=bool(args.edge_boost), canny1=args.canny1, canny2=args.canny2,
        crop_pad=args.crop_pad,
        manual_center=args.crop_center, crop_box_scale=args.crop_box_scale,
        panel_w=panel_w, panel_h=panel_h,
        darken_bg_amount=args.darken_bg,
        trail_stride=args.trail_stride
    )

    cv2.imwrite(args.output, film)
    print(f"Saved {args.output} ({film.shape[1]}x{film.shape[0]})")

if __name__ == "__main__":
    main()