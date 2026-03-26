import cv2
import os
import sys
import numpy as np

parent_dir = os.path.join(os.path.dirname(__file__),'..')
#上の行で造ったパスを整形して、プロジェクト全体から参照できるようにする
sys.path.append(os.path.normpath(parent_dir))

from core.config_loader import load_settings

def main():
    
    base_path = os.path.abspath(os.path.dirname(__file__))
    # そこから一つ上がってプロジェクトルートにする
    project_root = os.path.abspath(os.path.join(base_path, '..'))
    save_dir = os.path.join(project_root, "data", "calibration")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # --- カメラ初期化 ---
    settings = load_settings()
    
    left_id = settings["camera"]["left_index"]
    right_id = settings["camera"]["right_index"]

    cap_L = cv2.VideoCapture(left_id)
    cap_R = cv2.VideoCapture(right_id)

    if not cap_L.isOpened() or not cap_R.isOpened():
        print("エラー: 2台のカメラをオープンできません。")
        print("ls /dev/video* でデバイス番号を確認してください。")
        return   

    print(f"--- プレビュー撮影モード開始 ---")
    print(f"保存先: {save_dir}")
    print("操作: [S]キーで保存 / [q]で終了")

    count = 0
    try:
        while True:
            #cmd = input(f"[{count:02d}枚撮影済] 次を撮りますか？ (Enter:撮影 / q:終了): ")
            
            ret_L,image_L = cap_L.read()
            ret_R,image_R = cap_R.read()
            
            if not ret_L or not ret_R:
                break

            # プレビュー用に左右を連結して表示
            preview = np.hstack((image_L, image_R))
            # 少し縮小して表示（画面に収まりやすくするため）
            show_frame =cv2.resize(preview, (None, None), fx=0.8, fy=0.8)
            
            cv2.putText(show_frame, f"Saved: {count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Calibration Capture (L | R)", show_frame)

            key = cv2.waitKey(1) & 0xFF

            # 's' キーで保存
            if key == ord('s'):
                count += 1
                fname_L = os.path.join(save_dir, f"left{count:02d}.jpg")
                fname_R = os.path.join(save_dir, f"right{count:02d}.jpg")

                cv2.imwrite(fname_L,image_L)
                cv2.imwrite(fname_R,image_R)

                print(f" >> 保存成功: {os.path.basename(fname_L)} / {os.path.basename(fname_R)}")

            # 'q' キーで終了
            elif key == ord('q'):
                break
            
            else:
                print("  >> エラー: 画像の取得に失敗しました。")
        
    except KeyboardInterrupt:
        print("\n強制終了されました。")
    
    finally:
        cap_L.release()
        cap_R.release()
        print(f"--- 終了。合計 {count} セットの画像を保存しました ---")

if __name__ == "__main__":
    main() 