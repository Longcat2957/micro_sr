import torch
import math
import numpy as np
from typing import Union, Iterable
from datetime import datetime

def get_current_datetime() -> str:
    now = datetime.now()
    return now.strftime("%Y%m%d%H%M")

class AverageMeter:
    """
    평균값 및 표준편차를 계산하고 저장하는 클래스
    """
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """
        초기화 메서드
        """
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.var = 0
        self.std = 0

    def update(self, val: Union[torch.Tensor, int, float, Iterable], n: int = 1) -> None:
        """
        값을 입력받아 평균값과 표준편차를 계산하고 저장하는 메서드

        :param val: 입력값
        :param n: 입력값의 개수
        """
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu().item()

        if isinstance(val, Iterable):
            val = np.mean(val)

        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        if self.count > 1:
            self.var = ((self.count - 1) * self.var + (val - self.avg) ** 2) / self.count
            self.std = math.sqrt(self.var)