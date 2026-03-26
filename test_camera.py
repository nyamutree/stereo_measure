import cv2

def main():
    # eMeet C960を2台繋いでいる場合、通常は 0番 と 2番 に割り当てられます
    # (1番と3番はメタデータ用なので、映像は偶数番を使います)
    cam_l = cv2.VideoCapture(0)
    cam_r = cv2.VideoCapture(2)

    print("カメラを初期化しています...")

    if not cam_l.isOpened() or not cam_r.isOpened():
        print("エラー: 2台のカメラを同時にオープンできませんでした。")
        print("USBポートの差し込みを確認するか、ls /dev/video* を実行してください。")
        return

    # 1枚ずつ画像を読み込む
    ret_l, frame_l = cam_l.read()
    ret_r, frame_r = cam_r.read()

    if ret_l and ret_r:
        # 画像ファイルとして現在のフォルダに保存
        cv2.imwrite("left_test.jpg", frame_l)
        cv2.imwrite("right_test.jpg", frame_r)
        print("成功！ 画像を保存しました:")
        print("  - left_test.jpg")
        print("  - right_test.jpg")
    else:
        print("エラー: 画像の取得に失敗しました。")

    # リソースを解放
    cam_l.release()
    cam_r.release()

if __name__ == "__main__":
    main()
