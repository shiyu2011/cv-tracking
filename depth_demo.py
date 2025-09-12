# depth_demo.py
# two view geometry and depth demo

import argparse, os
import numpy as np
import cv2


#utility functinos
def load_intrinsic(calib_path):
    fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Failed to open file {calib_path}")
    
    K = fs.getNode("K").mat()
    D = fs.getNode("D").mat()
    baseline = fs.getNode("baseline").real()   # <-- read scalar
    fs.release()
    if K is None:
        raise RuntimeError("Intrinsic matrix K not found in the calibration file.")
    if D is None:
        D = np.zeros((5, 1))
    return K, D, baseline

def detect_and_match(img1, img2, max_feats=2000, ratio=0.75):
    orb = cv2.ORB_create(nfeatures=max_feats)
    k1, d1 = orb.detectAndCompute(img1, None)
    k2, d2 = orb.detectAndCompute(img2, None)
    if d1 is None or d2 is None:
        raise RuntimeError("No features detected in one of the images.")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(d1, d2, k=2)
    good = [m for m,n in knn if m.distance < ratio*n.distance]
    pts1 = np.float32([k1[m.queryIdx].pt for m in good])
    pts2 = np.float32([k2[m.trainIdx].pt for m in good])
    return pts1, pts2

def triangulate(P1, P2, pts1, pts2):
    pts1_h = pts1.T; pts2_h = pts2.T
    X_h = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    X = (X_h[:3] / X_h[3]).T
    return X
    
def reproj_error(P, X, pts):
    X_h = np.hstack([X, np.ones((len(X), 1))])
    proj = (P @ X_h.T).T
    proj = proj[:, 0:2] / proj[:, 2:3]
    return np.linalg.norm(proj - pts, axis=1)

def visualize(img, pts, depth, title=""):
    vis = img.copy()
    zmin, zmax = np.percentile(depth, [5, 95])
    norm = np.clip((depth - zmin) / (zmax - zmin), 0, 1)
    colors = cv2.applyColorMap((norm*255).astype(np.uint8), cv2.COLORMAP_JET)
    for (u, v), c in zip(pts, colors):
        cv2.circle(vis, (int(u), int(v)), 3, tuple(int(x) for x in c[0]), -1)
    cv2.putText(vis, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return vis


#stereo mode
def run_stereo(imgL, imgR, K, D, baseline):
    ptsL, ptsR = detect_and_match(imgL, imgR)
    F, mask = cv2.findFundamentalMat(ptsL, ptsR, cv2.FM_RANSAC, 1.0, 0.999)
    mask = mask.ravel().astype(bool)
    ptsL, ptsR = ptsL[mask], ptsR[mask]
    
    #epipolar error check
    
    #projection matrices (rectified assumption)
    P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))]) #left camera at origin as reference
    P2 = K @ np.hstack([np.eye(3), np.array([[-baseline,0,0]]).T]) #right camera translated along x axis by baseline
    
    #Triangulate
    X = triangulate(P1, P2, ptsL, ptsR)
    
    #errors
    e1 = reproj_error(P1, X, ptsL)
    e2 = reproj_error(P2, X, ptsR)
    reproj_err = (e1 + e2) / 2
    
    # Disparity depth check
    fx = K[0,0]
    disparity = ptsL[:,0] - ptsR[:,0]
    valid = disparity > 1e-3
    Z_disparity = fx * baseline / disparity[valid]
    mae = np.mean(np.abs(Z_disparity - X[valid,2])) if np.any(valid) else np.nan #this is only valid if it is rectified system
    
    vis = visualize(imgL, ptsL, X[:,2], title=f"Stereo Depth (MAE: {mae:.2f}, Reproj: {np.mean(reproj_err):.2f})")
    return vis

def run_mono(img1, img2, K, D):
    pts1, pts2 = detect_and_match(img1, img2)
    E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 1.0)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    pts1, pts2 = pts1[mask.ravel().astype(bool)], pts2[mask.ravel().astype(bool)]
    P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])
    P2 = K @ np.hstack([R, t])
    
    X = triangulate(P1, P2, pts1, pts2)
    e1 = reproj_error(P1, X, pts1)
    e2 = reproj_error(P2, X, pts2)
    reproj_err = (e1 + e2) / 2
    vis = visualize(img1, pts1, X[:,2], title=f"Mono Depth (Reproj: {np.mean(reproj_err):.2f})")
    return vis

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="stereo", choices=["stereo", "mono"], help="Mode: stereo or mono")
    ap.add_argument("--left", type=str, default="./playground_1l/im0.png", help="Left image path")
    ap.add_argument("--right", type=str, default="./playground_1l/im1.png", help="Right image path (for stereo and mono)")
    ap.add_argument("--calib", type=str, default="./playground_1l/calib.yaml", help="Calibration file path (YAML)")
    ap.add_argument("--show", default=False, action="store_true", help="Show the result")
    args = ap.parse_args()
    
    imgL = cv2.imread(args.left, cv2.IMREAD_COLOR)
    imgR = cv2.imread(args.right, cv2.IMREAD_COLOR)
    if imgL is None or imgR is None:
        raise FileNotFoundError(f"Failed to load images.")
    
    K, D, basline = load_intrinsic(args.calib)
    print(f"Loaded K:\n{K}\nD:\n{D}\nBaseline: {basline}")
    if args.mode == "stereo":
        if basline is None or basline <= 0:
            raise ValueError("Baseline must be provided for stereo mode.")
        vis = run_stereo(imgL, imgR, K, D, basline)
    else:
        vis = run_mono(imgL, imgR, K, D)
        
    if args.show:
        cv2.imshow("Depth Demo", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    #save the image
    out_path = f"depth_{args.mode}.png"
    cv2.imwrite(out_path, vis)
        
if __name__ == "__main__":
    main()
            