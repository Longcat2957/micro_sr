import torch
import cv2
import numpy as np
from model.base import load_model
from model.rlfn import RLFN
from model.safmn import SAFMN
from model.mobilesr import MOBILESR
from utils.io import sisr_preprocess, sisr_postprocess
import time
if __name__ == "__main__":
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net = RLFN().to(DEVICE)
    net = SAFMN(36).to(DEVICE)
    #net = MOBILESR().to(DEVICE)
    try:
        # net = load_model(net, "./test_weights/rlfn_final_400.pth")
        net = load_model(net, "./test_weights/SAFMN_x4_final_300.pth")
        #net = load_model(net, "./test_weights/mobilesr_final_200.pth")
    except:
        print(f"Failed to load model ... ")
    net = net.eval()

    # lr_tensor = sisr_preprocess("mobilesr_test_output.png").unsqueeze(0)
    lr_tensor = sisr_preprocess("./test_weights/ms3_01.png").unsqueeze(0)
    lr_tensor = lr_tensor.to(DEVICE)
    start = time.time()
    for i in range(100):
        with torch.no_grad():
            sr_tensor = net(lr_tensor)
    et = time.time() - start
    sr_np_arrs = sisr_postprocess(sr_tensor)
    img_np = sr_np_arrs[0]
    print(f"TIME = {et/100:.5f}")
    cv2.imwrite('mobilesr_test_output_x2.png', img_np)
    # cv2.imwrite('mobilesr_test_half_output.png', cv2.resize(img_np, (1280//2, 896//2)))