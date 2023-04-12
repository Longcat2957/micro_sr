import os
import random
import shutil
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str)
parser.add_argument('--dst', type=str, default='split')

def split_files_by_ratio(root_dir: str, ratio: float, dst: str):
    # 새로운 폴더 생성
    dst_root = os.path.join(os.path.dirname(os.path.abspath(root_dir)), dst)
    os.makedirs(dst_root, exist_ok=True)

    dst_train_root = os.path.join(dst_root, 'train')
    dst_valid_root = os.path.join(dst_root, 'valid')
    os.makedirs(dst_train_root, exist_ok=True)
    os.makedirs(dst_valid_root, exist_ok=True)

    # 디렉토리 내부 파일 리스트
    files = os.listdir(root_dir)
    num_files = len(files)

    # 비율에 맞게 분할할 인덱스 계산
    split_index = int(num_files * ratio)

    # 파일 리스트를 랜덤하게 섞어서 나누기
    random.shuffle(files)
    train_files = files[:split_index]
    valid_files = files[split_index:]

    # 파일 이동
    for filename in tqdm(train_files):
        src = os.path.join(root_dir, filename)
        dst = os.path.join(dst_train_root, filename)
        shutil.copy(src, dst)

    for filename in tqdm(valid_files):
        src = os.path.join(root_dir, filename)
        dst = os.path.join(dst_valid_root, filename)
        shutil.copy(src, dst)
    
    print(f"IMAGE SPLIT COMPLETE, TRAIN : {len(train_files)} / VALID : {len(valid_files)}")

if __name__ == '__main__':
    opt = parser.parse_args()
    src = opt.src
    dst = opt.dst
    split_files_by_ratio(src, 0.8, dst)