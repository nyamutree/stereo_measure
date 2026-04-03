import cv2
import numpy as np
import os
import sys

# 自作設定ローダーの読み込み
parent_dir = os.path.join(os.path.dirname(__file__),'..')
sys.path.append(os.path.normpath(parent_dir))
from core.config_loader import load_settings

def main():
    # 1. 設定とキャリブレーションデータの読み込み
    settings = load_settings()
    params_path = os.path.join(parent_dir, "config", "stereo_params.npz")

    if not os.path.exists(params_path):
        print("エラー: stereo_params.npz が見つかりません。先にキャリブレーションを行ってください。")
        return
    
    data = np.load(params_path)
    # 補正用マップの取得（映像を真っ直ぐにするための型紙）
    map_L1, map_L2 = data['mapL1'], data['mapL2']
    map_R1, map_R2 = data['mapR1'], data['mapR2']
    Q = data['Q']

    # 2. カメラの初期化
    cap_L = cv2.VideoCapture(settings["camera"]["left_index"])
    cap_R = cv2.VideoCapture(settings["camera"]["right_index"])

    # 3. 視差計算アルゴリズム (StereoBM) の設定
    # 左右の画像の「ズレ」を探すための処理の設定
    # cv2.StereoBM_create() <- 視差探索エンジン 左の画像の特徴が右の画像で何pixelずれているか
    # 探し出すアルゴリズム
    # numDisparities: どれだけ遠近の幅を探すか(16の倍数)
    # blockSize：どれくらいの大きさの塊で一致する場所を探すか(奇数で設定)
    stereo =  cv2.StereoBM_create(numDisparities=128, blockSize=15)

    print ("距離測定を開始します... [q]キーで終了") 

    while True:
        retL, frameL = cap_L.read()
        retR, frameR = cap_R.read()
        
        if not retL or not retR:break

        # 4. 映像の補正 (Remapping)
        # レンズの歪みを取り除き、左右の行を完璧に揃えます
        # cv2.ramap() <- 画像補正回路 ｷｬﾘﾌﾞﾚｰｼｮﾝのmapから歪んだ画像をピクセル単位で移動しまっすぐに
        imgL_rect = cv2.remap(frameL, map_L1, map_L2, cv2.INTER_LINEAR)
        imgR_rect = cv2.remap(frameR, map_R1, map_R2, cv2.INTER_LINEAR)

        # 視差計算のためにグレースケール化
        imgL_gray = cv2.cvtColor(imgL_rect, cv2.COLOR_BGR2GRAY)
        imgR_gray = cv2.cvtColor(imgR_rect, cv2.COLOR_BGR2GRAY)

        # 5. 視差 (Disparity) の計算
        disparity = stereo.compute(imgL_gray, imgR_gray)

        # 6. 距離 (Depth) への変換
        # Q行列を使って、視差(pixel)を実際の座標(mm)に変換します
        # cv2.reprojectImageTo3D() <- 3次元復元回路 視差の情報を実際の空間のXYZ座標に変換
        # Q行列：ｷｬﾘﾌﾞﾚｰｼｮﾝで算出した焦点距離、カメラ感が詰まった「変換テーブル」
        points_3D = cv2.reprojectImageTo3D(disparity, Q)

        # 画面中央 (中心点) の距離を取得してみる
        h, w = imgL_gray.shape
        s = 20
        # --- [安定化回路] 中心付近の21x21エリアの値を統計処理 ---
        center_region = points_3D[h//2-s : h//2+s, w//2-s : w//2+s, 2]
        #center_dist = points_3D[h//2, w//2, 2] / 10.0 # mm -> cm 変換 (// を使うと整数で計算)
        
        # 不正な値（0以下や無限大）を除外
        #volid_depths = center_region[(center_region > 0) & (center_region < 5000.0)]
        mask = (center_region > 0) & (center_region < 5000.0) & (np.isfinite(center_region))
        volid_depths = center_region[mask]

        if len(volid_depths) > 0:
            # 平均ではなく「中央値」を使うことで突発的なノイズを無視
            center_dist = np.median(volid_depths) / 10.0 #mm -> cm
        else :
            center_dist = 0.0


        # 結果の表示
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, 
                                 cv2.CV_8U)
        cv2.putText(imgL_rect, f"Distance:{center_dist:.1f}cm",(50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 画面中央に照準（赤十字）を表示
        cv2.drawMarker(imgL_rect, (w//2, h//2), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        
        cv2.imshow("Stereo Measure (Rectified)", imgL_rect)
        cv2.imshow("Disparity Map", disp_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap_L.release()
    cap_R.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()