# vis_bop.py — BlenderProc BOP 출력의 6D pose를 이미지에 3D 박스로 시각화
# 사용법: python vis_bop.py <scene폴더경로> [im_id]
#   예: python vis_bop.py output_test/bop_data/z_bracket/train_pbr/000000 6
import os, sys, json
import os.path as osp
import numpy as np
import cv2
from plyfile import PlyData

MODEL_PATH = "/home/eunbin/gitHub/gdrnpp_bop2022/datasets/BOP_DATASETS/z_bracket/models/obj_000001.ply"
OUT_DIR = "vis_bop_output"
os.makedirs(OUT_DIR, exist_ok=True)

# 3D 박스 12개 모서리
EDGES = [(0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),
         (0,4),(1,5),(2,6),(3,7)]

def load_model_bbox(path):
    """모델 ply의 3D 바운딩박스 8개 꼭짓점 (mm)"""
    ply = PlyData.read(path)
    v = ply["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], axis=1)
    mn, mx = pts.min(0), pts.max(0)
    corners = np.array([[x, y, z] for x in (mn[0], mx[0])
                                   for y in (mn[1], mx[1])
                                   for z in (mn[2], mx[2])], dtype=np.float64)
    return corners

def project(pts3d, R, t, K):
    cam = R @ pts3d.T + t.reshape(3, 1)
    proj = K @ cam
    proj = proj[:2] / proj[2]
    return proj.T

def main():
    scene_dir = sys.argv[1]
    filt_im = int(sys.argv[2]) if len(sys.argv) > 2 else None

    scene_gt = json.load(open(osp.join(scene_dir, "scene_gt.json")))
    scene_cam = json.load(open(osp.join(scene_dir, "scene_camera.json")))
    bbox3d = load_model_bbox(MODEL_PATH)

    # rgb 폴더 찾기 (jpg 또는 png)
    rgb_dir = osp.join(scene_dir, "rgb")

    count = 0
    for im_id_str in sorted(scene_gt.keys(), key=lambda x: int(x)):
        im_id = int(im_id_str)
        if filt_im is not None and im_id != filt_im:
            continue

        # 이미지 로드 (jpg 우선, 없으면 png)
        img_path = osp.join(rgb_dir, f"{im_id:06d}.jpg")
        if not osp.exists(img_path):
            img_path = osp.join(rgb_dir, f"{im_id:06d}.png")
        if not osp.exists(img_path):
            print(f"이미지 없음: {img_path}"); continue
        img = cv2.imread(img_path)

        K = np.array(scene_cam[im_id_str]["cam_K"]).reshape(3, 3)

        # 이 이미지의 모든 객체 인스턴스
        for inst in scene_gt[im_id_str]:
            R = np.array(inst["cam_R_m2c"]).reshape(3, 3)
            t = np.array(inst["cam_t_m2c"])
            corners2d = project(bbox3d, R, t, K).astype(int)
            for i, j in EDGES:
                cv2.line(img, tuple(corners2d[i]), tuple(corners2d[j]), (0, 255, 0), 2)
            # 중심에 라벨
            c = corners2d.mean(0).astype(int)
            cv2.putText(img, f"obj{inst['obj_id']}", tuple(c),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        out_path = osp.join(OUT_DIR, f"{osp.basename(scene_dir)}_im{im_id:06d}.png")
        cv2.imwrite(out_path, img)
        n_obj = len(scene_gt[im_id_str])
        print(f"저장: {out_path}  (객체 {n_obj}개)")
        count += 1
        if count >= 10:
            break

    print(f"\n총 {count}장 저장됨 → {OUT_DIR}/")

if __name__ == "__main__":
    main()