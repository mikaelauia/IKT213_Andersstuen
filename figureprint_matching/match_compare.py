import cv2
import numpy as np
import time
from pathlib import Path
import csv
import re

def load_gray(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read {path}")
    return img

def ratio_test(knn_matches, ratio=0.75):
    good = []
    for m, n in knn_matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

def ransac_inliers(kp1, kp2, matches, thresh=3.0):
    if len(matches) < 4:
        return 0, None, None
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, thresh)
    inliers = int(mask.sum()) if mask is not None else 0
    return inliers, H, mask

def draw_inlier_matches(img1, img2, kp1, kp2, matches, mask, max_draw=80):
    mdraw = min(max_draw, len(matches))
    matches_to_draw = matches[:mdraw]
    if mask is not None:
        mask_list = mask.ravel().tolist()[:mdraw]
    else:
        mask_list = None
    return cv2.drawMatches(img1, kp1, img2, kp2, matches_to_draw, None,
                           matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=mask_list,
                           flags=cv2.DrawMatchesFlags_DEFAULT)

def run_orb_bf(img1, img2, ratio=0.75):
    t0 = time.perf_counter()
    orb = cv2.ORB_create(nfeatures=3000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    t1 = time.perf_counter()
    if des1 is None or des2 is None:
        return {"name": "ORB+BF", "kps_1": len(kp1), "kps_2": len(kp2),
                "good": 0, "inliers": 0,
                "t_detect": t1-t0, "t_match": 0.0, "t_ransac": 0.0, "t_total": t1-t0,
                "kp1": kp1, "kp2": kp2, "matches": [], "mask": None}
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)
    good = ratio_test(knn, ratio)
    t2 = time.perf_counter()
    inliers, H, mask = ransac_inliers(kp1, kp2, good, thresh=3.0)
    t3 = time.perf_counter()
    return {"name": "ORB+BF",
            "kps_1": len(kp1), "kps_2": len(kp2),
            "good": len(good), "inliers": inliers,
            "t_detect": t1-t0, "t_match": t2-t1, "t_ransac": t3-t2, "t_total": t3-t0,
            "kp1": kp1, "kp2": kp2, "matches": good, "mask": mask}

def run_sift_flann(img1, img2, ratio=0.75):
    t0 = time.perf_counter()
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    t1 = time.perf_counter()
    if des1 is None or des2 is None:
        return {"name": "SIFT+FLANN", "kps_1": len(kp1), "kps_2": len(kp2),
                "good": 0, "inliers": 0,
                "t_detect": t1-t0, "t_match": 0.0, "t_ransac": 0.0, "t_total": t1-t0,
                "kp1": kp1, "kp2": kp2, "matches": [], "mask": None}
    des1 = des1.astype(np.float32)
    des2 = des2.astype(np.float32)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=64)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    knn = flann.knnMatch(des1, des2, k=2)
    good = ratio_test(knn, ratio)
    t2 = time.perf_counter()
    inliers, H, mask = ransac_inliers(kp1, kp2, good, thresh=3.0)
    t3 = time.perf_counter()
    return {"name": "SIFT+FLANN",
            "kps_1": len(kp1), "kps_2": len(kp2),
            "good": len(good), "inliers": inliers,
            "t_detect": t1-t0, "t_match": t2-t1, "t_ransac": t3-t2, "t_total": t3-t0,
            "kp1": kp1, "kp2": kp2, "matches": good, "mask": mask}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def find_two_images(folder: Path):
    imgs = sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])
    return imgs[0], imgs[1]

def numeric_sort_key(folder: Path):
    m = re.search(r"(\d+)$", folder.name)
    return int(m.group(1)) if m else 0

def compare_pair(img1_path: Path, img2_path: Path, out_dir: Path, set_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    img1 = load_gray(img1_path)
    img2 = load_gray(img2_path)
    results = []
    for fn in (run_orb_bf, run_sift_flann):
        res = fn(img1, img2, ratio=0.75)
        results.append(res)
        vis = draw_inlier_matches(img1, img2, res["kp1"], res["kp2"], res["matches"], res["mask"])
        cv2.imwrite(str(out_dir / f"{set_name}_{res['name'].replace('+','-')}.jpg"), vis)
    return results

def run_batch():
    root = Path(__file__).parent
    files_root = root / "files"
    output_root = root / "output"
    output_root.mkdir(exist_ok=True)
    csv_path = output_root / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "set_name","label","pipeline","kps_img1","kps_img2",
            "good_matches","inliers","inlier_ratio",
            "t_detect_s","t_match_s","t_ransac_s","t_total_s"
        ])
        for folder in sorted(files_root.iterdir(), key=numeric_sort_key):
            if not folder.is_dir():
                continue
            label = "same" if folder.name.startswith("same_") else ("different" if folder.name.startswith("different_") else "other")
            img1_path, img2_path = find_two_images(folder)
            set_name = folder.name
            print(f"Checking {set_name}...")
            out_dir = output_root / set_name
            results = compare_pair(img1_path, img2_path, out_dir, set_name)
            for r in results:
                inlier_ratio = (r["inliers"]/r["good"]) if r["good"]>0 else 0.0
                writer.writerow([
                    set_name, label, r["name"], r["kps_1"], r["kps_2"],
                    r["good"], r["inliers"], f"{inlier_ratio:.3f}",
                    f"{r['t_detect']:.4f}", f"{r['t_match']:.4f}", f"{r['t_ransac']:.4f}", f"{r['t_total']:.4f}"
                ])
        uia1 = root / "UiA front1.png"
        uia2 = root / "UiA front3.png"
        if not uia2.exists():
            uia2 = root / "UiA front3.jpg"
        if uia1.exists() and uia2.exists():
            print("Checking UiA_pair...")
            results = compare_pair(uia1, uia2, output_root / "UiA_pair", "UiA")
            for r in results:
                inlier_ratio = (r["inliers"]/r["good"]) if r["good"]>0 else 0.0
                writer.writerow([
                    "UiA_pair", "scene", r["name"], r["kps_1"], r["kps_2"],
                    r["good"], r["inliers"], f"{inlier_ratio:.3f}",
                    f"{r['t_detect']:.4f}", f"{r['t_match']:.4f}", f"{r['t_ransac']:.4f}", f"{r['t_total']:.4f}"
                ])

if __name__ == "__main__":
    run_batch()
