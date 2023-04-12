import os
import sys
import torch
import cv2
from PIL import Image
import numpy as np

def is_image_file(path: str) -> bool:
    """
    입력된 파일 경로의 확장자를 확인하여 이미지 파일인지 확인합니다.

    Args:
        path (str): 확인할 파일 경로.

    Returns:
        bool: 입력된 파일이 이미지 파일인 경우 True를 반환하고, 그렇지 않은 경우 False를 반환합니다.
    """
    # OpenCV로 읽을 수 있는 이미지 포맷의 확장자를 리스트로 선언합니다.
    image_extensions = [".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".webp", ".pbm", ".pgm", ".ppm", ".pxm",
                        ".pnm", ".pfm", ".sr", ".ras", ".tiff", ".tif", ".exr", ".hdr", ".pic"]

    # 파일 경로의 확장자를 추출합니다.
    ext = os.path.splitext(path)[-1].lower()

    # 추출한 확장자가 이미지 포맷 중 하나인지 확인합니다.
    if ext in image_extensions:
        return True
    else:
        return False

def read_img(p: str) -> np.ndarray:
    """
    이미지 파일을 읽어서 NumPy 배열로 반환합니다.

    Args:
        p (str): 읽어올 이미지 파일 경로.

    Returns:
        np.ndarray: 이미지 파일을 읽어온 후 변환된 NumPy 배열.

    Raises:
        TypeError: 입력이 str 타입이 아닌 경우 발생합니다.
        IOError: 이미지 파일을 읽어오지 못한 경우 발생합니다.
    """
    # 입력이 str 타입인지 검사합니다.
    if not isinstance(p, str):
        raise TypeError(f"입력은 str 타입이어야 합니다. 입력 타입: {type(p)}")

    try:
        # 이미지 파일을 RGB 색상공간으로 읽어옵니다.
        img = Image.open(p).convert('RGB')

        # 이미지를 NumPy 배열로 변환합니다.
        img = np.array(img)

        return img

    except IOError as e:
        # 이미지 파일을 읽어오지 못한 경우 에러 메시지를 출력하고 프로그램을 종료합니다.
        print(f"[ERROR]: {e}")
        sys.exit(-1)

def to_tensor(arr: np.ndarray) -> torch.Tensor:
    """
    입력된 NumPy 배열을 PyTorch tensor로 변환하고, 0~1 사이의 값을 갖도록 정규화합니다.

    Args:
        arr (numpy.ndarray): 변환할 NumPy 배열.

    Returns:
        torch.Tensor: 변환된 PyTorch tensor.

    Raises:
        TypeError: 입력이 numpy.ndarray가 아닌 경우 발생합니다.
        ValueError: 입력 배열의 shape이 (H, W, C) 또는 (B, H, W, C) 형태가 아닌 경우 발생합니다.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"입력은 numpy.ndarray 형태여야 합니다. 입력 타입: {type(arr)}")

    # 입력 배열의 shape이 (H, W, C) 또는 (B, H, W, C) 형태인지 확인합니다.
    if len(arr.shape) != 3 and len(arr.shape) != 4:
        raise ValueError(f"입력 배열의 shape은 (H, W, C) 또는 (B, H, W, C) 형태여야 합니다. 입력 shape: {arr.shape}")

    # 배열 값을 0~1 사이로 정규화합니다.
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)

    # PyTorch tensor로 변환합니다.
    if len(arr.shape) == 3:
        arr = np.transpose(arr, (2, 0, 1))
    else:
        arr = np.transpose(arr, (0, 3, 1, 2))

    return torch.from_numpy(arr)

def sisr_preprocess(p:str) -> torch.Tensor:
    """
    Super Resolution 입력 이미지를 전처리합니다.

    Args:
        p (str): 입력 이미지 파일 경로.

    Returns:
        torch.Tensor: 전처리된 이미지를 담은 PyTorch tensor.

    Raises:
        TypeError: 입력이 str 타입이 아닌 경우 발생합니다.
    """
    if not isinstance(p, str):
        raise TypeError(f"입력은 str 타입이어야 합니다. 입력 타입: {type(p)}")

    # 입력 이미지 파일을 NumPy 배열로 읽어옵니다.
    arr = read_img(p)

    # NumPy 배열을 PyTorch tensor로 변환합니다.
    tensor = to_tensor(arr)

    return tensor

def sisr_postprocess(t: torch.Tensor) -> list:
    """
    Super Resolution 결과로 나온 PyTorch tensor를 후처리합니다.

    Args:
        t (torch.Tensor): Super Resolution 결과로 나온 PyTorch tensor.

    Returns:
        list: 후처리된 결과물을 담은 NumPy 배열의 리스트.

    Raises:
        NotImplementedError: 입력 tensor의 차원 수가 3 또는 4가 아닌 경우 발생합니다.
    """
    # tensor의 차원 수를 확인합니다.
    if len(t.size()) != 4:
        raise NotImplementedError(f"SIZE ERROR = {len(t.size())}")

    # tensor의 값 범위를 0~1로 제한합니다.
    t = t.clamp_(0.0, 1.0)

    # tensor의 값 범위를 0~255로 변환합니다.
    t *= 255.0

    # tensor를 반올림(round)하고, dtype을 uint8로 변경합니다.
    o = t.round().to(dtype=torch.uint8).cpu().numpy()

    # numpy.ndarray의 차원 순서를 변경하여 반환합니다.
    return [np.transpose(o[i], axes=[1, 2, 0])[:,:,::-1] for i in range(len(o))]