# Task Vector Prediction

초기 미세조정 과정의 파라미터 변화로 이후 task vector의 궤적을 예측할 수 있는지 살펴보는 연구 코드입니다. CLIP 비전 인코더의 full fine-tuning과 LoRA를 비교하고, 관측한 변화량에 지수 함수의 합을 맞춰 예측 모델을 평가합니다.

## 연구와 구현

심현성이 KAIRI 학부생 연구인턴(공식 참여 기간 2025년 7–10월)과 이후 개인 연구에서 진행한 실험입니다. 문제·가정 검토, 학습 및 분석 코드 통합, 서버 실험과 결과 해석에 생성형 AI의 도움을 사용했습니다.

- `train.py`: YAML 설정 기반 CLIP 미세조정, 초기 가중치·분류 텍스트 특징·task vector·체크포인트 저장
- `analyze.py`: 초기 관측점 선택, 지수 궤적 적합, 예측 가중치 평가와 시각화
- `src/tvp/predictor.py`: `tau(t) = sum_i A_i * (1 - exp(-r_i*t))`, `r_i > 0`
- `src/tvp/analyzer.py`: A의 해석적 해와 rate의 gradient 업데이트를 교대로 최적화
- `configs/`: ViT-B/32·ViT-L/14, CIFAR-10·EuroSAT·Food-101, SGD·momentum·AdamW, full·LoRA 조합

LoRA adapter 공간과 `B @ A`로 만든 operational 공간은 서로 다릅니다. 현재 operational 변환은 LoRA alpha/r scaling을 포함하지 않으므로 실제 merge된 가중치와 동일한 수치라고 해석하지 않습니다. 특정 정확도 개선이나 미세조정 대체 효과가 재현되었다고 주장하지 않습니다.

## 환경

기존 `requirements.txt`는 Linux/CUDA 12.1에서 수집한 환경 목록이며 Windows/CPU용 공통 설치 파일이 아닙니다. Python 3.10–3.12와 장치에 맞는 PyTorch·torchvision 조합을 먼저 준비한 뒤 다음 핵심 의존성을 설치합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
```

Windows에서는 `.venv\Scripts\activate`를 사용합니다. 실험에는 Hugging Face 모델·데이터 다운로드와 충분한 메모리가 필요합니다. W&B는 `logging.wandb=false`로 끌 수 있습니다.

## 실행

```bash
python train.py --config configs/vitb32_eurosat_adamw_lora.yaml --set logging.wandb=false
python analyze.py --exp_dir outputs/<설정명>/<실행ID> --prediction_space adapter --k 3 --N 6
```

학습 결과는 `outputs/<설정명과 override>/<실행ID>/` 아래에 저장됩니다. 분석에는 `effective_config.yaml`, `theta0.pt`, `text_features.pt`, `task_vectors/tau_*.pt`, `tau_star.pt`가 필요합니다. 설정의 `save_tau_every_n_steps`와 optimizer별 간격에 맞는 N개의 체크포인트가 있어야 합니다.

```bash
python train.py --config configs/vitb32_eurosat_adamw_lora.yaml --set logging.wandb=false --resume_id <기존실행ID>
```

재개할 때는 기존과 동일한 config·override를 사용합니다. W&B를 꺼도 원래 실행 폴더를 다시 사용하며 초기 기준점과 텍스트 특징을 보존합니다. 필요한 파일이 없거나 config가 달라지면 새 학습으로 조용히 넘어가지 않고 오류를 냅니다.

## 검증과 결과의 범위

```bash
python -m unittest discover -s tests -v
```

CPU 합성 테스트로 비기본 체크포인트 저장 간격, 부족하거나 중복되는 적합 관측점, 재개 파일 보존, 지수 궤적의 경계값과 작은 rate의 수치 안정성을 검증합니다. 이 검증은 CLIP 재학습이나 실제 데이터에서 예측 성능을 재현한 결과가 아닙니다. GPU 실험, LoRA scaling·초기 adapter 기준점 및 후보 모델 선택의 전체 분석은 추가 확인 대상입니다.

대용량 checkpoint, W&B 실행 기록, 다운로드 데이터와 연구실 행정·공유 자료는 저장소에 포함하지 않습니다.

공식 참여 기간은 [KAIRI 인턴 참여확인서 원본](https://github.com/hyeongus2/hyeongus2/blob/main/docs/certificates/KAIRI-internship.pdf)에서 확인할 수 있습니다.
