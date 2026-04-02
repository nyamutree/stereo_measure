import cv2
import os
import sys
import glob        
import subprocess  
import subprocess # 外部プログラムを実行するために必要
import numpy as np
from flask import Flask, Response, render_template, jsonify

parent_dir = os.path.join(os.path.dirname(__file__),'..')
#上の行で造ったパスを整形して、プロジェクト全体から参照できるようにする
sys.path.append(os.path.normpath(parent_dir))

from core.config_loader import load_settings


#サーバーとして動く宣言
app = Flask(__name__, template_folder=os.path.join(parent_dir, 'templates'))

#設定の読み込み
settings = load_settings()
left_id = settings["camera"]["left_index"]
right_id = settings["camera"]["right_index"]

#画像の保存先の設定
base_path = os.path.abspath(os.path.dirname(__file__))
# そこから一つ上がってプロジェクトルートにする
project_root = os.path.abspath(os.path.join(base_path, '..'))
save_dir = os.path.join(project_root, "data", "calibration")

print(f"DEBUG: 画像を確認するディレクトリ -> {save_dir}")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#カメラの初期化
cap_L = cv2.VideoCapture(left_id)
cap_R = cv2.VideoCapture(right_id)

count = 0

#web画面用のHTML

# URLの登録
@app.route('/')
def index():
    # templates/index.html を読み込んでスマホに送る
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            ret_L,img_L = cap_L.read()
            ret_R,img_R = cap_R.read()

            if not (ret_L and ret_R):
                break

            combined =  np.hstack((img_L, img_R))
            #プレビュー用にリサイズ
            preview = cv2.resize(combined,(640, int(combined.shape[0]*640 / combined.shape[1])))
            _, buffer = cv2.imencode('.jpg',preview,[cv2.IMWRITE_JPEG_QUALITY, 40])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save')
def save():
    global count 
    ret_L,img_L = cap_L.read()
    ret_R,img_R = cap_R.read()

    if ret_L and ret_R:
        count += 1
        cv2.imwrite(os.path.join(save_dir, f"left{count:02d}.jpg"),img_L)
        cv2.imwrite(os.path.join(save_dir, f"right{count:02d}.jpg"),img_R)
        return f"保存成功！ ({count}セット目)"
    return "エラー：撮影失敗"


#def main():
#    
#    base_path = os.path.abspath(os.path.dirname(__file__))
#    # そこから一つ上がってプロジェクトルートにする
#    project_root = os.path.abspath(os.path.join(base_path, '..'))
#    save_dir = os.path.join(project_root, "data", "calibration")
#
#    if not os.path.exists(save_dir):
#        os.makedirs(save_dir)
#
#    # --- カメラ初期化 ---
#    settings = load_settings()
#    
#    left_id = settings["camera"]["left_index"]
#    right_id = settings["camera"]["right_index"]
#
#    cap_L = cv2.VideoCapture(left_id)
#    cap_R = cv2.VideoCapture(right_id)



#    if not cap_L.isOpened() or not cap_R.isOpened():
#        print("エラー: 2台のカメラをオープンできません。")
#        print("ls /dev/video* でデバイス番号を確認してください。")
#        return   
#
#    print(f"--- プレビュー撮影モード開始 ---")
#    print(f"保存先: {save_dir}")
#    print("操作: [S]キーで保存 / [q]で終了")
#
#    count = 0
#    try:
#        while True:
#            #cmd = input(f"[{count:02d}枚撮影済] 次を撮りますか？ (Enter:撮影 / q:終了): ")
#            
#            ret_L,image_L = cap_L.read()
#            ret_R,image_R = cap_R.read()
#            
#            if not ret_L or not ret_R:
#                break
#
#            # プレビュー用に左右を連結して表示
#            preview = np.hstack((image_L, image_R))

#            # 下部に黒いバーを作成 (高さ50ピクセル、幅は画像と同じ)
#            bar_height = 50
#            status_bar = np.zeros((bar_height,preview.shape[1],3),dtype=np.uint8)

            # バーに文字を書き込む
#            text = f"Saved: {count:02d} | [S]: Save  [Q]: Quit"
#            cv2.putText(status_bar, text, (20, 35), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            #画像とバーを垂直に連結
#            combined = np.vstack((preview, status_bar))

#            # 横幅を 1000px くらいに固定すると、VNCでも見やすくなります
#            target_width = 1000 
#            scale = target_width / combined.shape[1]
#            target_height = int(combined.shape[0] * scale)
#            
#            show_frame = cv2.resize(combined, (target_width, target_height))
#            
#            cv2.imshow("Calibration Capture (L | R)", show_frame)


#            # 少し縮小して表示（画面に収まりやすくするため）
#            ##show_frame =cv2.resize(preview, (None, None), fx=0.8, fy=0.8)
#   
#            ##cv2.imshow("Calibration Capture (L | R)", show_frame)
#
#            key = cv2.waitKey(1) & 0xFF

#            # 's' キーで保存
#            if key == ord('s'):
#                count += 1
#                fname_L = os.path.join(save_dir, f"left{count:02d}.jpg")
#                fname_R = os.path.join(save_dir, f"right{count:02d}.jpg")
#
#                cv2.imwrite(fname_L,image_L)
#                cv2.imwrite(fname_R,image_R)
#
#                print(f" >> 保存成功: {os.path.basename(fname_L)} / {os.path.basename(fname_R)}")
#
#            # 'q' キーで終了
#            elif key == ord('q'):
#                break
#            
#            #else:
#            #    print("  >> エラー: 画像の取得に失敗しました。")
#        
#    except KeyboardInterrupt:
#        print("\n強制終了されました。")
#    
#    finally:
#        cap_L.release()
#        cap_R.release()
#        print(f"--- 終了。合計 {count} セットの画像を保存しました ---")

@app.route('/get_count')
def get_count():
    """現在保存されている画像のペア数を返す"""
    # left*.jpgのリストを取得して数を数える
    left_imgs = glob.glob(os.path.join(save_dir,"left*.jpg"))
    return jsonify({"count":len(left_imgs)})

@app.route('/run_calibration')
def run_caribration():
    """キャリブレーション計算スクリプトを実行する"""
    # 安全のためにサーバーでも枚数を確認する
    # glob.glob() <- 指定したフォルダ内から該当のファイルをすべて探し出し、リストにする
    left_imgs = glob.glob(os.path.join(save_dir, "left*.jpg"))

    if len(left_imgs) < 20:
        # jsonify(...) <-JavaScriptが理解しやすい「JSON」と形式に変換
        return jsonify({"status": "error", "message": f"画像が足りません（現在{len(left_imgs)}枚）"})
    
    try:
        # 計算スクリプト（calibrate_stereo.py）のパスを指定
        # utils フォルダ内にある想定です
        script_path = os.path.join(parent_dir,"utils","calibrate_stereo.py")

        # 外部プロセスとして実行し、完了を待つ
        # capture_output=True にすることで、エラーが出た場合にその内容を取得できます
        # ["python3", script_path] <- ターミナルで python3 utils/calibrate_stereo.py とするときと同じ指示をラズパイに出す
        # capture_output=True <- 計算プログラムで画面に出力したログを変数の中に保存
        # text=True　<- 保存したログを人が読めるテキストとして扱う
        result = subprocess.run(["python3", script_path],capture_output=True,text=True)

        if result.returncode == 0:
            return jsonify({"status": "success", "message": "キャリブレーションが完了しました！"})
        else:
            return jsonify({"status": "error", "message": result.stderr})

    except Exception as e:
        return jsonify({"status":"error", "massage":str(e)})




if __name__ == "__main__":
    try:
        app.run(host='0.0.0.0',port=5000,threaded=True)
    finally:
        cap_L.release()
        cap_R.release()