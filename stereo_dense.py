#!/usr/bin/env python3
import argparse, os
import numpy as np
import cv2

def load_intrinsics(calib_path):
    fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Cannot open calib file {calib_path}")

    def read_first(*keys):
        for k in keys:
            node = fs.getNode(k)
            if not node.empty():
                return node.mat()
        return None

    K = read_first("K", "camera_matrix", "intrinsic", "CameraMatrix")
    D = read_first("D", "distortion_coefficients", "distCoeffs", "Distortion")
    baseline = fs.getNode("baseline").real()   # <-- read scalar
    fs.release()
    if K is None:
        raise RuntimeError("No camera matrix found in calib file")
    if D is None:
        D = np.zeros((1,5), np.float64)
        

    return K.astype(np.float64), D.astype(np.float64), baseline

def build_sgbm(ndisp=None, min_disp=0, block_size=5):
    # numDisparities must be multiple of 16; choose a sensible default if not given
    if ndisp is None or ndisp < 16:
        ndisp = 128
    num_disp = int(np.ceil(ndisp / 16.0) * 16)

    P1 = 8 * 3 * block_size**2
    P2 = 32 * 3 * block_size**2
    sgbm = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=P1, P2=P2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    return sgbm, num_disp

def disparity_to_depth(disparity, fx, baseline_m, doffs=0.0):
    # disparity from SGBM is fixed-point (scaled by 16)
    disp = disparity.astype(np.float32) / 16.0
    disp = disp - float(doffs)
    # avoid div-by-zero / negatives
    valid = disp > 0.1
    depth = np.zeros_like(disp, dtype=np.float32)
    depth[valid] = (fx * baseline_m) / disp[valid]
    return depth, valid

def colorize_depth(depth, valid_mask, clip=(0.5, 30.0)):
    d = depth.copy()
    d[~valid_mask] = np.nan
    # clip to a nice range (meters) for visualization
    d_vis = np.clip(d, clip[0], clip[1])
    # normalize ignoring NaNs
    mn = np.nanpercentile(d_vis, 5)
    mx = np.nanpercentile(d_vis, 95)
    norm = (d_vis - mn) / (mx - mn + 1e-6)
    norm = np.nan_to_num(norm)
    cm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    # paint invalid as black
    cm[~valid_mask] = (0, 0, 0)
    return cm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", type=str, default="./playground_1l/im0.png", help="Left image path")
    ap.add_argument("--right", type=str, default="./playground_1l/im1.png", help="Right image path (for stereo and mono)")
    ap.add_argument("--calib", type=str, default="./playground_1l/calib.yaml", help="Calibration file path (YAML)")
    ap.add_argument("--doffs", type=float, default=0.0, help="disparity offset")
    ap.add_argument("--ndisp", type=int, default=None, help="max disparity (e.g., 941 for ETH3D)")
    ap.add_argument("--save_disp", default="disply.png")
    ap.add_argument("--save_depth", default="True_depth.exr")
    ap.add_argument("--save_depth_color", default="depth_color.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    # load images
    L = cv2.imread(args.left, cv2.IMREAD_COLOR)
    R = cv2.imread(args.right, cv2.IMREAD_COLOR)
    if L is None or R is None:
        raise FileNotFoundError("Could not load left/right images")

    # load intrinsics
    K, D, baseline = load_intrinsics(args.calib)
    fx = float(K[0,0])

    # Optional: convert to grayscale (SGBM works on grayscale)
    Lg = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
    Rg = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)

    # Build matcher and compute disparity
    sgbm, num_disp = build_sgbm(ndisp=args.ndisp, min_disp=0, block_size=5)
    disp = sgbm.compute(Lg, Rg)  # CV_16S, scaled by 16

    if args.save_disp:
        # Save a normalized disparity visualization
        disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(args.save_disp, disp_vis.astype(np.uint8))

    # Convert disparity -> depth (meters)
    depth_m, valid = disparity_to_depth(disp, fx=fx, baseline_m=baseline, doffs=args.doffs)

    # if args.save_depth:
    #     # Save 32-bit float EXR/PNG (PNG will quantize); EXR preserves floats if supported
    #     ext = os.path.splitext(args.save_depth)[1].lower()
    #     if ext == ".exr":
    #         cv2.imwrite(args.save_depth, depth_m)
    #     else:
    #         # scale for PNG; store invalid as 0
    #         dm = depth_m.copy()
    #         dm[~valid] = 0.0
    #         cv2.imwrite(args.save_depth, (dm * 1000.0).astype(np.uint16))  # millimeters

    depth_color = colorize_depth(depth_m, valid_mask=valid, clip=(0.5, 50.0))
    cv2.imwrite(args.save_depth_color, depth_color)

    if args.show:
        cv2.imshow("Left", L)
        cv2.imshow("Disparity (scaled)", (disp.astype(np.float32)/16.0 / (num_disp or 1) * 255).astype(np.uint8))
        cv2.imshow("Depth color (m)", depth_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("Done. fx=%.2f, baseline=%.5f m, numDisp=%d" % (fx, args.baseline, num_disp))

if __name__ == "__main__":
    main()
