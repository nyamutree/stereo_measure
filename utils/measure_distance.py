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

    # --- 【重要】ここを追加：補正後のパラメータを取得 ---
    P1 = data['P1'] # 補正後の投影行列
    T = data['T']# カメラ間の移動ベクトル

    Q = data['Q']

    fx_rect = P1[0,0]           # 補正後の焦点距離 (f)
    # B_calc = np.linalg.norm(T)  # キャリブレーション上のカメラ間隔 (B)
    B_calc = float(abs(T[0][0]))



    # 2. カメラの初期化
    cap_L = cv2.VideoCapture(settings["camera"]["left_index"])
    cap_R = cv2.VideoCapture(settings["camera"]["right_index"])

    # 3. 視差計算アルゴリズム (StereoBM) の設定
    # 左右の画像の「ズレ」を探すための処理の設定
    # cv2.StereoSGBM_create() <- 視差探索エンジン 左の画像の特徴が右の画像で何pixelずれているか
    # 探し出すアルゴリズム
    # numDisparities: どれだけ遠近の幅を探すか(16の倍数)
    # blockSize：どれくらいの大きさの塊で一致する場所を探すか(奇数で設定)
    blockSize = (settings["Stereo"]["blockSize"])
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities = 160,        # 16の倍数
        blockSize = blockSize ,          # 3～11くらい
        P1 = 8 * 3 * blockSize**2,          # パラメータ（このままでOK）
        P2 = 32 * 3 * blockSize**2,         # パラメータ（このままでOK）
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100, 
        speckleRange = 32,
        mode = cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    print ("距離測定を開始します... [q]キーで終了") 

    # MOG2 というアルゴリズムを使う背景差分処理の呼び出し
    # MOG2 <-「影」を自動で判別して無視したり、少しだけ風で揺れているカーテンなどを背景として学習
    # history=500: 500フレーム分を背景学習に使う
    # varThreshold=50: 値が大きいほど変化に鈍感（ノイズに強く）なる
    # detectShadows=True: 影を検出して、測定から除外する
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)


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
        disparity = stereo.compute(imgL_gray, imgR_gray).astype(np.float32)/16.0

        # ---物体検出とサイズ推定 ---
        # 背景差分を実行して「動いた部分」を白くする
        fgmask = fgbg.apply(imgL_rect)

        # ノイズを消す（小さな白い点々を消して、形を整える）
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        # 白い塊を探す
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        target_box = None #見つけた枠を保存する関数

        if contours:
            # 画面内で一番面積が大きい塊を1つ選ぶ
            best_cnt = max(contours, key = cv2.contourArea)

            # あまりに小さい（500ピクセル以下）ものはゴミとして無視
            if(cv2.contourArea(best_cnt)>500):
                # 塊を囲む四角形（x座標, y座標, 幅w, 高さh）を計算
                target_box = cv2.boundingRect(best_cnt)
            
        # ここから、見つかった枠(target_box)を使って距離とサイズを出す
        if target_box is not None:
            x, y, w, h = target_box
            cx, cy = x + w//2, y + h//2 # 物体の中心点 

        # 6. 距離 (Depth) への変換
        # 画面中央 (中心点) の距離を取得してみる
        # h, w = imgL_gray.shape
        # s = 20
        # --- [安定化回路] 中心付近の21x21エリアの値を統計処理 ---
        # center_region_disp = disparity[h//2-s : h//2+s, w//2-s : w//2+s] # (// を使うと整数で計算)
        
        # 不正な値（0以下や無限大）を除外
        # valid_disp = center_region_disp[(center_region_disp > 1) & (center_region_disp < 200)]

        # 中心付近の視差を取得
        d = disparity[cy, cx] 
        
        #if len(valid_disp ) > 0:
        if d > 0 :
            # 平均ではなく「中央値」を使うことで突発的なノイズを無視
            # --- ここを追加：抽出した有効な視差から中央値(d)を決める ---
            #d = np.median(valid_disp)
            # 物理式： Z = (f * B) / d
            dist_mm = (fx_rect * B_calc) / d
            Z_cm = dist_mm / 10.0

            # --- 【ここが肝】サイズ推定の計算 ---
            # 実寸(mm) = (ピクセル幅 * 距離Z) / 焦点距離f
            real_w_mm = (w * dist_mm) / fx_rect
            real_h_mm = (h * dist_mm) / fx_rect

            # 画面に緑の枠を描く
            cv2.rectangle(imgL_rect, (x, y), (x + w, y + h), (0,255, 0), 2)
            # 結果を文字で表示
            label = (f"Z:{Z_cm:.1f}cm W{real_w_mm/10.0:.1f}cm H{real_h_mm/10:.1f}cm")
            cv2.putText(imgL_rect, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # デバッグ用にターミナルに数値を出す
            print(f"d: {d:.2f}px, f: {fx_rect:.1f}, B: {B_calc:.1f}mm -> Z: {Z_cm:.1f}cm")
        else :
            Z_cm = 0.0
            real_w_mm = 0.0
            real_h_mm = 0.0


        # 結果の表示
        disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, 
                                 cv2.CV_8U)
        #cv2.putText(imgL_rect, f"Distance:{Z_cm:.1f}cm",(50,50),
        #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 画面中央に照準（赤十字）を表示
        #cv2.drawMarker(imgL_rect, (w//2, h//2), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        
        cv2.imshow("Stereo Measure - Object Detection", imgL_rect)
        cv2.imshow("Disparity Map", disp_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap_L.release()
    cap_R.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()