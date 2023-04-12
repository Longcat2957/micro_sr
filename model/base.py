import torch
import torch.nn as nn
from thop import profile

def save_model(model: nn.Module, path: str):
    # 모델의 상태 저장
    state_dict = model.state_dict()
    
    # 저장할 디렉토리와 파일 이름 지정
    
    # 모델 저장
    torch.save(state_dict, path)
    
    # print(f"모델이 {path}에 저장되었습니다.")

'''
    try:
        loaded_model = load_model(MyModel(), "mymodel")
    except ValueError as e:
        print(e)
        # 처리할 작업
    except RuntimeError as e:
        print(e)
        # 처리할 작업
    else:
        # 모델 사용
'''

def load_model(model: nn.Module, path: str):

    try:
        # 모델 파일 불러오기
        state_dict = torch.load(path)
    except FileNotFoundError:
        raise ValueError(f"{path} 파일이 없습니다.")
    except:
        raise RuntimeError(f"{path} 파일을 불러올 수 없습니다.")
    
    # 모델 클래스 인스턴스화
    try:
        model.load_state_dict(state_dict)
    except:
        raise RuntimeError("모델 클래스와 state_dict 형식이 일치하지 않습니다.")
    
    print(f"{path} 파일에서 모델을 불러왔습니다.")
    return model

def evaluate_model(model: nn.Module, input_size: tuple = (1, 3, 256, 256)):
    # 모델의 FLOPs와 파라미터 수 계산
    flops, params = profile(model, inputs=(torch.randn(*input_size).cuda(),))
    
    # 결과 출력
    print(f"모델의 FLOPs: {flops / (10. ** 9):.3f} GFLOPs")
    print(f"모델의 파라미터 수: {params / (10. ** 6):.2f} M")