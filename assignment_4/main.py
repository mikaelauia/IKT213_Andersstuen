from pathlib import Path
import cv2
import numpy as np

def harris_corner_detection(image_path: Path, save_path: Path):
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    out = img.copy()
    out[dst > 0.02 * dst.max()] = [0, 0, 255]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), out)

def prep_gray(img_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    return g

def edge_mask(img_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    e = cv2.Canny(g, 60, 180)
    k = np.ones((3, 3), np.uint8)
    e = cv2.dilate(e, k, iterations=1)
    e = cv2.morphologyEx(e, cv2.MORPH_CLOSE, k, iterations=1)
    return e

def estimate_homography(pts1: np.ndarray, pts2: np.ndarray):
    method = getattr(cv2, "USAC_MAGSAC", None)
    if method is not None:
        return cv2.findHomography(pts1, pts2, method, ransacReprojThreshold=3.0,
                                  confidence=0.999, maxIters=10000)
    return cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=3.0, confidence=0.999)

def is_homography_reasonable(H: np.ndarray, w: int, h: int) -> bool:
    if H is None:
        return False
    corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    x, y = warped[:, 0], warped[:, 1]
    A = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if A < 0.1 * (w * h) or A > 4.0 * (w * h):
        return False
    v = np.roll(warped, -1, axis=0) - warped
    v_next = np.roll(v, -1, axis=0)
    cross_z = v[:, 0] * v_next[:, 1] - v[:, 1] * v_next[:, 0]
    return np.all(cross_z > 0) or np.all(cross_z < 0)

def align_with_sift(image_to_align_path: Path, reference_image_path: Path):
    im1 = cv2.imread(str(image_to_align_path))
    im2 = cv2.imread(str(reference_image_path))
    g2 = prep_gray(im2)
    mask2 = edge_mask(im2)
    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.02, edgeThreshold=10, sigma=1.2)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=200))
    scales = [0.5, 0.75, 1.0, 1.25, 1.5]
    best, best_inliers = None, 0
    for s in scales:
        im1s = cv2.resize(im1, dsize=None, fx=s, fy=s,
                          interpolation=cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC)
        g1 = prep_gray(im1s)
        kp1, des1 = sift.detectAndCompute(g1, None)
        kp2, des2 = sift.detectAndCompute(g2, mask2)
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            continue
        knn = flann.knnMatch(des1, des2, k=2)
        good = [m for m, n in knn if m.distance < 0.8 * n.distance]
        if len(good) < 4:
            continue
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        Hs, inliers = estimate_homography(pts1, pts2)
        if Hs is None or inliers is None:
            continue
        inl = int(inliers.sum())
        if inl > best_inliers and is_homography_reasonable(Hs, im2.shape[1], im2.shape[0]):
            best = (Hs, s, kp1, kp2, good, inliers, im1s)
            best_inliers = inl
    Hs, s, kp1, kp2, good, inliers, im1s = best
    H = Hs @ np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)
    h, w = im2.shape[:2]
    aligned = cv2.warpPerspective(im1, H, (w, h))
    matches = cv2.drawMatches(im1s, kp1, im2, kp2, good, None,
                              matchesMask=inliers.ravel().tolist(),
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return aligned, matches

def main():
    base = Path(__file__).resolve().parent
    in_dir = base / "Input"
    out_dir = base / "output"
    ref_path = in_dir / "reference_img.png"
    align_path = in_dir / "align_this.jpg"
    out_dir.mkdir(parents=True, exist_ok=True)
    harris_corner_detection(ref_path, out_dir / "harris.png")
    aligned_img, matches_img = align_with_sift(align_path, ref_path)
    cv2.imwrite(str(out_dir / "aligned.png"), aligned_img)
    cv2.imwrite(str(out_dir / "matches.png"), matches_img)

if __name__ == "__main__":
    main()
