import cv2
import sys
import os
import numpy as np

#今実行しているプログラムファイルの場所からそのファイルが入っているフォルダのパスを導き一つ上の階層に移動
parent_dir = os.path.join(os.path.dirname(__file__),'..')
#上の行で造ったパスを整形して、プロジェクト全体から参照できるようにする
sys.path.append(os.path.normpath(parent_dir))

from  core.config_loader import load_settings

def main():
    settings = load_settings()
    left_id = settings["camera"]["left_index"]
    right_id = settings["camera"]["right_index"]
    width = settings["camera"]["width"]
    height = settings["camera"]["height"]

    cap_L = cv2.VideoCapture(left_id)
    cap_R = cv2.VideoCapture(right_id)

    for cap in [cap_L, cap_R]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)

    print("プレビューを開始します。'q'キーで終了します。")

    while True:
        #左右同時に画像を読み込む
        ret_L, frame_L = cap_L.read()
        ret_R,frame_R = cap_R.read()

        if not ret_L or not ret_R:
            print("カメラから映像が取得できません")
            break
        
        #【重要】左右の画像を横につなげる
        combined_frame = cv2.hconcat([frame_L,frame_R])
        display_frame = cv2.resize(combined_frame, (640, 240))
        
        cv2.imshow("Stereo Camera Test",display_frame)

        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
    
    cap_L.release()
    cap_R.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

        

