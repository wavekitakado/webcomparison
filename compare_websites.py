import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import csv

def process_urls(csv_file):
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        url_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for i, url_pair in enumerate(url_reader):
            url1, url2 = url_pair
            img1 = take_screenshot(url1)
            img2 = take_screenshot(url2)

            combined_diff_img = compare_images(img1, img2)
            combined_diff_img.save(f"combined_diff_{i}.png")


def take_screenshot(url):
    options = Options()
    options.headless = True
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--log-level=3")
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
    driver.set_window_size(1920, 1080)
    driver.get(url)
    
    scroll_height = driver.execute_script("return document.body.scrollHeight")
    driver.set_window_size(1920, scroll_height)
    
    png = driver.get_screenshot_as_png()
    img = Image.open(BytesIO(png))
    
    driver.quit()
    return img

def compare_images(img1, img2):
    # 画像のサイズを揃える
    width = max(img1.width, img2.width)
    height = max(img1.height, img2.height)

    img1_resized = img1.resize((width, height), Image.ANTIALIAS)
    img2_resized = img2.resize((width, height), Image.ANTIALIAS)

    img1 = cv2.cvtColor(np.array(img1_resized), cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(np.array(img2_resized), cv2.COLOR_RGB2BGR)

    # グレースケールに変換
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 差分を計算
    diff = cv2.absdiff(img1_gray, img2_gray)

    # 差分を二値化
    thresh = 65
    _, diff_bin = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)

    # 輪郭を検出して赤い矩形を描画
    contours, _ = cv2.findContours(diff_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 差分を強調表示した画像を一つにまとめる
    img1_diff_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_diff_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    combined_diff = np.hstack((img1_diff_rgb, img2_diff_rgb))
    combined_diff_img = Image.fromarray(combined_diff)

    return combined_diff_img

if __name__ == "__main__":
    csv_file = "urls.csv"
    process_urls(csv_file)

