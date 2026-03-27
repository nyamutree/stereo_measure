import cv2
import numpy as np
import os
import glob

parent_dir = os.path.join(os.path.dirname(__file__),'..')
img_dir =  os.path.join(parent_dir,"data","calibration")
save_path = os.path.join(parent_dir,"config", "stereo_params.npz")

# --- チェッカーボードの設定 (お手持ちのボードに合わせて修正してください) ---
PATTERN_SIZE = (10, 7)  # 交点の数（列, 行）
SQUARE_SIZE = 15.0      # マスの実サイズ (mm)

# 3D空間での理想的な点座標を作成(ゆがみを計算するための基準（チェッカーボードの形について）を作成)

# チェッカーボードの「交点の数」と同じ分だけ、3D座標の座標（0,0,0）を用意
objp = np.zeros((PATTERN_SIZE[0]*PATTERN_SIZE[1],3),np.float32)

# 上で作った0だった座標に**「正しい位置（理想の座標）」**を書き込み
# mgrid を使って、(0,0),(1,0),(2,0)... と規則正しい連番を振る
# SQUARE_SIZE（15mm）を掛けることで、**「0番目の点は(0mm, 0mm)、隣の点は(15mm, 0mm)...」**という風に、現実世界のミリ単位の座標を完成
objp[:,:2] = np.mgrid[0:PATTERN_SIZE[0],0:PATTERN_SIZE[1]].T.reshape(-1,2)*SQUARE_SIZE

# 現実世界での正しい座標（上で作った objp）」を、成功した枚数分だけ保存
objpoints = [] #3D_Point

# 「カメラの画像の中で、角が何ピクセル目に見えたか」という、実際に写真に写った座標を保存
img_Points_L = [] #2D_Point_L
img_Points_R = [] #2D_Point_R

# 保存された画像を取得
L_imgs = sorted(glob.glob(os.path.join(img_dir,"left*.jpg")))
R_imgs = sorted(glob.glob(os.path.join(img_dir,"right*.jpg")))

print(f"{len(L_imgs)} 組の画像を処理中...")

for f_l, f_r in zip(L_imgs,R_imgs):
    img_L = cv2.imread(f_l)
    img_R = cv2.imread(f_r)

    gray_L = cv2.cvtColor(img_L,cv2.COLOR_RGB2GRAY)
    gray_R = cv2.cvtColor(img_R,cv2.COLOR_RGB2GRAY)

    ret_L,corner_L = cv2.findChessboardCorners(gray_L,PATTERN_SIZE,None)
    ret_R,corner_R = cv2.findChessboardCorners(gray_R,PATTERN_SIZE,None)

    if ret_L and ret_R:
        objpoints.append(objp)
        img_Points_L.append(corner_L)
        img_Points_R.append(corner_R)
        print(f"✅ 成功: {os.path.basename(f_l)}")

# --- 計算開始 ---
print("計算中...（数分かかる場合があります）")
# 各カメラの単体キャリブレーション
ret_L, mtx_L, dist_L, rvecs_L, tvecs_L = cv2.calibrateCamera(objpoints, img_Points_L, gray_L.shape[::-1],None,None)
ret_R, mtx_R, dist_R, rvecs_R, tvecs_R = cv2.calibrateCamera(objpoints, img_Points_R, gray_R.shape[::-1],None,None)

# ステレオキャリブレーション（左右の位置関係を計算)
flags = cv2.CALIB_FIX_INTRINSIC
crireria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TermCriteria_EPS, 30, 1e-6)

ret, m1, d1, m2, d2, R, T, E, F =cv2.stereoCalibrate(
    objpoints, img_Points_L, img_Points_R, mtx_L, dist_L, mtx_R, dist_R, 
    gray_L.shape[::-1], criteria=crireria, flags=flags
)

# 結果を保存
np.savez(save_path, mtx_L=m1, dist_L=d1, mtx_R=m2, dist_R=d2, R=R, T=T )
print(f"🎉 完了！設定ファイルを保存しました: {save_path}")