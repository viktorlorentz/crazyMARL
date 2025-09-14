#!/usr/bin/env python3
# pipeline_multiframe.py
import argparse, cv2, numpy as np

def iter_frames(path, every=10, max_frames=200, resize_width=0, rotate="ccw"):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    frames, i = [], 0
    rot_mode = {"none": None, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "cw": cv2.ROTATE_90_CLOCKWISE}[rotate]
    while True:
        ok = cap.grab()
        if not ok: break
        if i % every == 0:
            ok, f = cap.retrieve()
            if not ok: break
            if rot_mode is not None:
                f = cv2.rotate(f, rot_mode)  # step 1: rotate at ingest (CCW default)
            if resize_width and resize_width > 0:
                h, w = f.shape[:2]
                f = cv2.resize(f, (resize_width, int(h * (resize_width / w))), interpolation=cv2.INTER_AREA)
            frames.append(f)
            if max_frames and len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    return frames

def build_background(frames, bg_samples=50):
    take = frames[:max(1, min(bg_samples, len(frames)))]
    return np.median(np.stack(take, axis=0), axis=0).astype(np.uint8)  # step 2

def apply_white_top(bg, frac=0.5):
    """White-out the top frac of the background, keep lower part intact (step 3 & 4)."""
    out = bg.copy()
    h = out.shape[0]
    cut = int(h * frac)
    out[:cut, :] = (255, 255, 255)  # true white
    return out

def motion_mask(frame, bg, blur_sigma=1.5, thresh=20, open_ksize=3, dilate_iter=1, min_area=0):
    # abs diff → grayscale → blur → threshold → optional morphology
    diff = cv2.absdiff(frame, bg)
    g = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    if blur_sigma > 0:
        g = cv2.GaussianBlur(g, (0, 0), blur_sigma)
    _, m = cv2.threshold(g, thresh, 255, cv2.THRESH_BINARY)
    if open_ksize and open_ksize > 1:
        k = np.ones((open_ksize, open_ksize), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    if dilate_iter > 0:
        m = cv2.dilate(m, None, iterations=dilate_iter)
    if min_area and min_area > 0:
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        keep = np.zeros_like(m)
        for c in cnts:
            if cv2.contourArea(c) >= min_area:
                cv2.drawContours(keep, [c], -1, 255, thickness=cv2.FILLED)
        m = keep
    return m

def composite_diffs(bg_white_top, frames, bg_ref, order="earliest", **mask_kw):
    """
    Step 5: Paste only diffs (hard overwrite) onto the prepared background (white top + real lower half).
    bg_ref is the *median background before whitening* for stable differencing.
    """
    out = bg_white_top.copy()
    seq = frames if order == "earliest" else list(frames)[::-1]
    for f in seq:
        m = motion_mask(f, bg_ref, **mask_kw)
        m3 = cv2.merge([m, m, m])
        out = np.where(m3 == 255, f, out)  # hard replace, no opacity
    return out

def main():
    ap = argparse.ArgumentParser(description="Rotate CCW, white top background, keep lower half, then paste motion diffs.")
    ap.add_argument("video", help="Path to input video")
    ap.add_argument("-o","--output", default="composite.png", help="Output image path")
    ap.add_argument("--every", type=int, default=10, help="Use every Nth frame")
    ap.add_argument("--max_frames", type=int, default=200, help="Max sampled frames")
    ap.add_argument("--bg_samples", type=int, default=50, help="Frames for median background")
    ap.add_argument("--resize", type=int, default=0, help="Resize width (0 = no resize)")
    ap.add_argument("--rotate", choices=["none","ccw","cw"], default="cw", help="Rotate at ingest (default: ccw)")
    ap.add_argument("--white_top_frac", type=float, default=0.5, help="Top fraction to white-out (0..1)")
    ap.add_argument("--thresh", type=int, default=20, help="Motion threshold")
    ap.add_argument("--blur_sigma", type=float, default=1.5, help="Gaussian blur sigma before threshold")
    ap.add_argument("--open", dest="open_ksize", type=int, default=3, help="Opening kernel size (0/1 to disable)")
    ap.add_argument("--dilate_iter", type=int, default=1, help="Dilate iterations (connects mask parts)")
    ap.add_argument("--min_area", type=int, default=0, help="Filter blobs smaller than this area (px)")
    ap.add_argument("--order", choices=["earliest","latest"], default="earliest", help="Overwrite order for diffs")
    args = ap.parse_args()

    frames = iter_frames(args.video, every=args.every, max_frames=args.max_frames,
                         resize_width=args.resize, rotate=args.rotate)
    if len(frames) < 2:
        raise SystemExit("Not enough frames. Try smaller --every or a longer video.")

    bg_ref = build_background(frames, bg_samples=args.bg_samples)            # clean median background (for differencing)
    bg_white = apply_white_top(bg_ref, frac=args.white_top_frac)             # top white, bottom real background

    out = composite_diffs(
        bg_white, frames, bg_ref,
        order=args.order,
        blur_sigma=args.blur_sigma, thresh=args.thresh,
        open_ksize=args.open_ksize, dilate_iter=args.dilate_iter, min_area=args.min_area
    )
    cv2.imwrite(args.output, out)
    print("Saved", args.output)

if __name__ == "__main__":
    main()