import cv2
import os

def save_background(frame,side):
"""カメラの画像を背景用として保存する"""
	file_path = f"calibrations/bg_{side}.jpg"
	cv2.imwrite(file_path, frame)
	print(f"背景画像を保存しました:"{file_path}")
	return file_path
