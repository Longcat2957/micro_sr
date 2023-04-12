import os
import cv2
import torch
import random
import numpy as np
import albumentations as A
from tqdm import tqdm
from multiprocessing import Pool
from typing import Tuple, Union, List
from torch.utils.data import Dataset
from .io import is_image_file, read_img, to_tensor

def get_interpolation():
    """
    임의의 cv2 interpolation 기법을 리턴하는 함수입니다.
    """
    interpolation = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4])
    return interpolation

class base_Dataset(Dataset):
    def __init__(self, root: str, preload: bool) -> None:
        super().__init__()

        # 이미지 디렉토리가 비어있는지 확인합니다.
        assert len(os.listdir(root)) > 0, f"이미지 디렉토리:{root} 가 비어있습니다."

        # 이미지 파일 경로를 리스트에 저장합니다.
        self.img_paths = [os.path.join(root, name) for name in os.listdir(root) if is_image_file(name)]

        # 이미지 파일이 없는 경우 예외를 발생시킵니다.
        assert len(self.img_paths) > 0, f"이미지 디렉토리:{root}에 이미지 파일이 없습니다."

        # 데이터셋 크기를 저장합니다.
        self.data_length = len(self.img_paths)

        # preload 인자가 True인 경우 데이터를 미리 로드합니다.
        if preload:
            self.datas = self._preload()
        else:
            self.datas = self.img_paths

    def __len__(self) -> int:
        return self.data_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def _preload(self) -> Union[List[torch.Tensor], torch.Tensor]:
        print(f"START IMAGE PRELOAD ...")
        with Pool(processes=os.cpu_count()//2) as pool:
            preloaded_data = []
            for img_data in tqdm(pool.imap_unordered(read_img, self.img_paths), total=self.data_length):
                preloaded_data.append(img_data)
        pool.close()
        pool.join()
        print("\033[F\033[J ==> IMAGE PRELOAD COMPLETE")
        return preloaded_data
    
class train_Dataset(base_Dataset):
    def __init__(self, root:str, preload:bool, patch:int, patch_size:int, hr:int, lr:int) -> None:
        super().__init__(root=root, preload=preload)
        self.patch = patch
        self.patch_size = patch_size

        # HR(Ground Truth) size, LR size
        self.hr = hr
        self.lr = lr

        # Albumentation Transform 객체 생성
        # Ground Truth -> HR obj(np.ndarray)
        self.gt2hr = A.Compose([
            A.HueSaturationValue(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=self.hr, width=self.hr)
        ])
        # HR obj -> LR obj(np.ndarray)
        self.hr2lr = A.Compose([
            A.AdvancedBlur(p=0.5),
            A.Downscale(scale_min=0.7, scale_max=0.95, p=0.5, interpolation=cv2.INTER_LINEAR),
            A.OneOf([A.GaussNoise(), A.ISONoise()], p=0.5),
            A.ImageCompression(quality_lower=80, quality_upper=100, p=0.5),
            A.AdvancedBlur(p=0.5),
            A.Resize(height=self.lr, width=self.lr),
            A.OneOf([A.GaussNoise(), A.ISONoise()], p=0.5),
            A.ImageCompression(quality_lower=80, quality_upper=100, p=0.5)
        ])

    def _apply_batched_imgs(self, t, batched_imgs:list) -> list:
        outputs = []
        for i in batched_imgs:
            outputs.append(t(image=i)['image'])
        return outputs

    def _get_image_patches(self, f:Union[str, np.ndarray]) -> list:
        # 이미지를 preload하지 않았을 경우 read_img 함수를 통해 np.ndarray 형태로 가져옵니다.
        if isinstance(f, str):
            img = read_img(f)
        else:
            img = f

        # 이미지에서 patch 숫자만큼 self.patch_size 크기의 이미지를 Random Cropping 합니다.
        img_patches = []
        for i in range(self.patch):
            h, w, _ = img.shape
            x = np.random.randint(0, h - self.patch_size)
            y = np.random.randint(0, w - self.patch_size)
            patch = img[x:x+self.patch_size, y:y+self.patch_size]
            img_patches.append(patch)

        # 이미지 패치를 (B, H, W, C) 형태로 변환합니다.
        # img_patches = np.stack(img_patches)

        return img_patches

    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor]:
        original_image = self.datas[idx]
        img_patches = self._get_image_patches(original_image)

        # Albumentation Transform 적용
        hr_obj = self._apply_batched_imgs(self.gt2hr, img_patches)
        lr_obj = self._apply_batched_imgs(self.hr2lr, hr_obj)

        # list to np.ndarray
        hr_obj = np.stack(hr_obj)
        lr_obj = np.stack(lr_obj)

        # np.ndarray -> torch.Tensor 변환
        hr_tensor = to_tensor(hr_obj)
        lr_tensor = to_tensor(lr_obj)

        return lr_tensor, hr_tensor
    @staticmethod
    def collate_fn(batch):
        """
        배치 단위로 데이터를 묶어주는 함수입니다.
        HR 이미지와 LR 이미지를 묶어서 튜플로 반환합니다.
        """
        hr_imgs, lr_imgs = zip(*batch)
        hr_imgs = torch.cat(hr_imgs, 0)
        lr_imgs = torch.cat(lr_imgs, 0)
        return hr_imgs, lr_imgs
    
class valid_Dataset(base_Dataset):
    def __init__(self, root:str, preload:bool, hr:int, lr:int) -> None:
        super().__init__(root=root, preload=preload)
    
        # HR(Ground Truth) size, LR size
        self.hr = hr
        self.lr = lr

        # Albumentation Transform 객체 생성
        # Ground Truth -> HR obj(np.ndarray)
        self.gt2hr = A.Compose([
            A.CenterCrop(height=self.hr, width=self.hr)
        ])
        # HR obj -> LR obj(np.ndarray)
        self.hr2lr = A.Compose([
            A.Resize(height=self.lr, width=self.lr, interpolation=cv2.INTER_LINEAR)
        ])

    def __getitem__(self, idx:int) -> Tuple[torch.Tensor, torch.Tensor]:
        original_image = self.datas[idx]
        # 이미지를 preload하지 않았을 경우 read_img 함수를 통해 np.ndarray 형태로 가져옵니다.
        if isinstance(original_image, str):
            gt_obj = read_img(original_image)
        else:
            gt_obj = original_image

        # HR 이미지 크기로 crop 후 resize
        hr_obj = self.gt2hr(image=gt_obj)['image']
        # HR 이미지를 LR 이미지 크기로 resize
        lr_obj = self.hr2lr(image=hr_obj)['image']

        hr_tensor = to_tensor(hr_obj)
        lr_tensor = to_tensor(lr_obj)
        
        return lr_tensor, hr_tensor
    
    @staticmethod
    def collate_fn(batch):
        """
        배치 단위로 데이터를 묶어주는 함수입니다.
        HR 이미지와 LR 이미지를 묶어서 튜플로 반환합니다.
        """
        hr_imgs, lr_imgs = zip(*batch)
        hr_imgs = torch.cat(hr_imgs, 0)
        lr_imgs = torch.cat(lr_imgs, 0)
        return hr_imgs, lr_imgs