import yaml
import os

def load_settings():
    # ファイルの場所を指定
    #config_path = "config/setting.yaml"
    base_path = os.path.dirname(__file__)
    config_path = os.path.normpath(os.path.jon(base_path,"../config/setting.yaml"))

    # 指定したパスのファイルを「読み込みモード('r')」で開き、中身を読み取ります
    with open(config_path, 'r',encoding='utf-8') as f:
        conf = yaml.safe_load(f)

        return conf 

if __name__ == "__main__":
    try:
        settings = load_settings()
         # ここで中身を表示してみたい！
        left_id = settings["camera"]["left_index"]
        right_id = settings["camera"]["right_index"]
        print("設定を読み込みました。")
        print(f"左カメラID: {left_id}")
        print(f"右カメラID: {right_id}")
        # 文字列の中に変数を埋め込む「f-string」という書き方を使っています
        print(f"解像度: {settings['camera']['width']}x{settings['camera']['height']}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")


