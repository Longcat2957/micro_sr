import sys
import os
import torch
import argparse
from tqdm import tqdm
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torch.utils.data import DataLoader
from model.base import save_model, load_model
from utils.dataset import train_Dataset, valid_Dataset
from utils.metric import AverageMeter, get_current_datetime

# amp 관련
from torch.cuda.amp import GradScaler, autocast

# 모델 관련 임포트
from model.scsrn import SCSRN, convert_model

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
# 모델학습 공통사항
# 학습데이터 관련 hyperparameter
parser.add_argument('--batch_size', type=int, default=4, help='배치 크기')
parser.add_argument('--root_dir', type=str, default='../data/DF2K_SPLIT', help='데이터셋 루트 디렉토리 경로')
parser.add_argument('--preload', action='store_true', help='데이터셋 미리 로드 여부')
parser.add_argument('--patch', type=int, default=4, help='패치의 갯수')
parser.add_argument('--patch_size', type=int, default=512, help='패치의 크기')
parser.add_argument('--hr_size', type=int, default=256, help='고해상도 이미지 크기')
parser.add_argument('--lr_size', type=int, default=64, help='저해상도 이미지 크기')

# 훈련 관련 hyperparameter
parser.add_argument("--amp", action='store_true', help="Automatic Mixed Precision")                             # <=== in progress
parser.add_argument("--num_workers", type=int, default=None, help="DataLoader num of workers")
parser.add_argument("--lr", type=float, default=1e-3, help="초기 Learning rate")
parser.add_argument("--loss", type=str, default="l1", choices=['l1', 'l2'], help="loss function to use")
parser.add_argument('--epochs', type=int, default=50, help='훈련 에포크 수')

# 모델 저장 관련 hyperparameter
parser.add_argument('--save_freq', type=int, default=10, help='모델 저장 빈도')
parser.add_argument('--tag', type=str, default=None)

# 모델 로딩 관련 hyperparameter
parser.add_argument('--load', type=str, default=None, help="pre-trained weights")

def main():
    opt = parser.parse_args()
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    start_time = get_current_datetime()
    upscale_ratio = opt.hr_size // opt.lr_size
    
    # Automatic Mixed Precision
    scaler = GradScaler() if opt.amp else None

    # 데이터셋 준비
    train_datas_root = os.path.join(opt.root_dir, 'train')
    valid_datas_root = os.path.join(opt.root_dir, 'valid')

    train_dataset = train_Dataset(train_datas_root, opt.preload, opt.patch, opt.patch_size,
                                  opt.hr_size, opt.lr_size)
    valid_dataset = valid_Dataset(valid_datas_root, opt.preload, opt.hr_size, opt.lr_size)

    actual_batch_size = opt.batch_size*opt.patch
    if opt.num_workers is None:
        data_loading_workers = min(os.cpu_count(), actual_batch_size)
    else:
        data_loading_workers = opt.num_workers
    train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True,
                              collate_fn=train_dataset.collate_fn, num_workers=data_loading_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=actual_batch_size, shuffle=True, num_workers=data_loading_workers)

    # 모델 준비
    net = SCSRN(upscale_ratio=upscale_ratio)
    
    # 모델 로드
    if opt.load is not None:
        try:
            net = load_model(net, opt.load)
            print(f"# Model load Success")
        except:
            print(f"# Model load failed ...")
    net = net.to(DEVICE)

    # 손실 함수
    if opt.loss == 'l1':
        criterion = torch.nn.L1Loss()
    elif opt.loss == 'l2':
        criterion = torch.nn.MSELoss()

    # 최적화
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)

    # 스케줄러
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10
    )

    # Metric
    train_loss_meter = AverageMeter()
    validation_loss_meter = AverageMeter()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    for e in range(1, opt.epochs+1):
        # Train
        net = net.train()
        train_bar = tqdm(train_loader, ncols=120)
        for lr, hr in train_bar:
            # load tensors to DEVICE
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            optimizer.zero_grad()

            # if use amp
            if opt.amp:
                # forward pass
                with autocast():
                    sr = net(lr)
                    loss = criterion(sr, hr)
                if torch.isnan(loss):
                    print(f"#[ERROR] loss is NAN")
                    continue
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss_meter.update(loss, 1)
                train_bar.set_description(
                    f"# TRAIN [{e}/{opt.epochs}] loss_avg = {train_loss_meter.avg:.5f}"
                )

            # if not use amp
            else:
                # forward pass
                sr = net(lr)
                loss = criterion(sr, hr)
                if torch.isnan(loss):
                    print(f"#[ERROR] loss is NAN")
                    continue
                loss.backward()
                optimizer.step()
                train_loss_meter.update(loss, 1)
                train_bar.set_description(
                    f"# TRAIN [{e}/{opt.epochs}] loss_avg = {train_loss_meter.avg:.5f}"
                )
        # Reset train loss meter
        train_loss_meter.reset()
        
        # reparameterization
        # eval_net = convert_model(net)
        # eval_net = eval_net.eval()
        net = net.eval()
        # Validation
        valid_bar = tqdm(valid_loader)
        for lr, hr in valid_bar:
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            with torch.no_grad():
                sr = net(lr)
            validation_loss = criterion(sr, hr)

            validation_loss_meter.update(validation_loss)

            psnr_score = peak_signal_noise_ratio(sr, hr)
            ssim_score = structural_similarity_index_measure(sr, hr)
            psnr_meter.update(psnr_score)
            ssim_meter.update(ssim_score)

            valid_bar.set_description(
                f"# VALID [{e}/{opt.epochs}] PSNR={psnr_meter.avg:.5f} SSIM={ssim_meter.avg:.5f}"
            )


        lr_scheduler.step(validation_loss_meter.avg)
        
        # validation에 사용된 Metric 초기화
        validation_loss_meter.reset()
        psnr_meter.reset()
        ssim_meter.reset()

        # 모델 저장 (Save freq에 도달했을 경우)
        if e % opt.save_freq == 0:
            # weight 파일의 부모 디렉토리
            WEIGHT_ROOT = "./weights"
            if not os.path.exists(WEIGHT_ROOT):
                os.makedirs(WEIGHT_ROOT)
            
            # 세부 디렉토리
            SAVE_ROOT = os.path.join(WEIGHT_ROOT, start_time)
            if not os.path.exists(SAVE_ROOT):
                os.makedirs(SAVE_ROOT)
            
            # save file name
            if opt.tag is not None:
                save_file_name = f"{opt.tag}_x{upscale_ratio}_{e}_{opt.epochs}.pth"
            else:
                save_file_name = f"{e}_x{upscale_ratio}_{opt.epochs}.pth"
            save_file_name = os.path.join(SAVE_ROOT, save_file_name)
            save_model(net, save_file_name)

        # 모델 저장 (마지막 이포크에 도달했을 경우)
        if e == opt.epochs:
            # 마지막 이포크를 마친 뒤
            # weight 파일의 부모 디렉토리
            WEIGHT_ROOT = "./weights"
            if not os.path.exists(WEIGHT_ROOT):
                os.makedirs(WEIGHT_ROOT)
            
            # 세부 디렉토리
            SAVE_ROOT = os.path.join(WEIGHT_ROOT, start_time)
            if not os.path.exists(SAVE_ROOT):
                os.makedirs(SAVE_ROOT)
            
            # save file name
            if opt.tag is not None:
                save_file_name = f"{opt.tag}_x{upscale_ratio}_final_{opt.epochs}.pth"
            else:
                save_file_name = f"final_x{upscale_ratio}_{opt.epochs}.pth"
            save_file_name = os.path.join(SAVE_ROOT, save_file_name)
            save_model(net, save_file_name)
            
        # del eval_net


if __name__ == '__main__':
    main()
    sys.exit(0)