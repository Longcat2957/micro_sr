import torch
from model.base import load_model, evaluate_model
from utils.io import sisr_preprocess, sisr_postprocess
from tools.export_onnx import export_onnx_model
from model.rlfn import RLFN
from model.mobilesr import MOBILESR

if __name__ == "__main__":
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # net = RLFN(upscale_ratio=2).to(DEVICE)
    # net = SAFMN(36).to(DEVICE)
    net = MOBILESR(upscale_ratio=2).to(DEVICE)
    # try:
    #     # net = load_model(net, "./test_weights/rlfn_final_400.pth")
    #     # net = load_model(net, "./test_weights/SAFMN_x4_final_300.pth")
    #     net = load_model(net, "./test_weights/mobilesr_final_200.pth")
    # except:
    #     print(f"Failed to load model ... ")
    net = net.eval()
    evaluate_model(net, (1, 3, 1440//2, 2560//2))
    export_onnx_model(net, (1, 3, 1440//2, 2560//2), 'mobilesr_x2_xx.onnx')