<div align="center">

<h1>RealView</h1>

<h3>일론 머스크가 공개한 테슬라 차량 화면(Reality View) 구조를 그대로 코드로 만들었습니다.<br>코드와 가중치를 모두 공개하고, 맥북 한 대로 1시간 46분 학습했습니다.</h3>

<p>
<a href="#빠른-시작"><img alt="명령 세 줄로 실행" src="https://img.shields.io/badge/%EC%8B%A4%ED%96%89-%EB%AA%85%EB%A0%B9%203%EC%A4%84-0f172a?style=for-the-badge"></a>
<a href="https://github.com/azet-llc/realview/releases/tag/v0.1.0"><img alt="가중치 받기" src="https://img.shields.io/badge/%EA%B0%80%EC%A4%91%EC%B9%98-v0.1.0%20%C2%B7%2066%20MB-16a34a?style=for-the-badge"></a>
</p>

<p><a href="README.md">English</a> | <b>한국어</b></p>

<img src="assets/realview.gif" width="860" alt="학습에 쓰지 않은 주행 세 화면: 카메라, 인코더 특징값만으로 같은 도로를 그린 RealView, 밤 주행 특징값을 넣은 RealView">

<sub>왼쪽: 카메라. 가운데: 인코더 특징값만 보고 같은 도로를 그린 RealView. 오른쪽: 같은 모델에 밤 주행의 특징값을 넣은 결과. 두 주행 모두 학습에 쓰지 않았습니다.</sub>

</div>

<br>

| 그 주행의 특징값을 줬을 때 | 다른 주행의 특징값을 줬을 때 | M4 Max 속도 | 모델 |
|---|---|---|---|
| 학습에 없던 주행에서 같은 순간을 **63.8%** 맞힘 | **1.5%** (무작위 2%) | 초당 **86프레임**, 8프레임마다 계산 한 번 | **3,300만** 파라미터, 66 MB |

## 소식

- **2026-11-27** v0.1.0 공개: 코드, 가중치(step 10,000 EMA), 학습에 쓰지 않은 60초 샘플 주행, 아래 수치 전부.
- **2026-11-22** 일론 머스크가 차량 화면을 만드는 방법을 [공개](https://x.com/elonmusk/status/2124458557525397780).

## 무엇인가

가운데 화면을 그리는 생성 모델은 카메라 화면을 보지 않습니다. 고정된 비전 인코더(DINOv2)가 그 프레임에서 뽑은 특징값과,
자기가 직전에 그린 화면만 받습니다. 그것만으로 도로를 다시 그립니다.

머스크의 글은 짧고 구체적입니다. 비전 인코더 특징값 → 별도의 확산 모델이 직전 생성 화면에 노이즈를 섞어 디노이즈 →
인코더 한 번에 8프레임 묶음. 이걸 `git clone` 해서 바로 돌릴 수 있는 저장소로 만들었습니다.

> 비공식 구현입니다. 테슬라와 관계가 없고, 테슬라의 코드·데이터·가중치를 쓰지 않았습니다.

## 빠른 시작

```sh
git clone https://github.com/azet-llc/realview && cd realview
pip install -r requirements.txt
sh scripts/download_weights.sh     # 생성 모델, TAESD(MIT), 60초 샘플 주행을 v0.1.0 릴리스에서 받음
python -m realview.run --video weights/sample_day.mp4 --out out.mp4
```

`out.mp4` 왼쪽은 카메라, 오른쪽은 RealView입니다. 내 블랙박스 영상(전방)을 넣어도 됩니다. 4:3으로 가운데를 자르고
256×192, 20 fps로 맞춥니다. 장치는 CUDA → Apple MPS → CPU 순서로 자동 선택합니다. MPS와 CPU에서 확인했고
(M4 Max CPU로 60초 샘플이 5분 15초), CUDA는 아직 확인하지 않았습니다.

```sh
python -m realview.run --video a.mp4 --feats zero              # 특징값을 0으로
python -m realview.run --video a.mp4 --feats shuffle:b.mp4     # 다른 주행의 특징값을 보여 줌
```

돌아갔다면 Star를 눌러 주세요. 다음에 공개로 만들 두 가지(다중 카메라 입력, 위에서 내려다본 지도 화면)를 정하는 데 씁니다.

## 갤러리

모두 학습에서 뺀 두 주행입니다. 빈 화면에서 시작한 실행의 4~12초 구간이며, 8프레임마다 계산 한 번입니다.

<table>
<tr>
<td align="center"><img src="assets/gallery/day.gif" width="420"><br><sub><b>낮 고속도로.</b> 이 클립은 100% 맞힘.</sub></td>
<td align="center"><img src="assets/gallery/curve.gif" width="420"><br><sub><b>차가 많은 구간.</b> 65%. 가까운 차가 생겼다 사라집니다.</sub></td>
</tr>
<tr>
<td align="center"><img src="assets/gallery/night.gif" width="420"><br><sub><b>밤.</b> 이 클립 46%, 밤 주행 전체 26%. 가장 약한 부분.</sub></td>
<td align="center"><img src="assets/gallery/blind.gif" width="420"><br><sub><b>특징값 0.</b> 같은 모델, 같은 시작인데 도로가 나타나지 않습니다.</sub></td>
</tr>
</table>

## 구조

<p align="center"><img src="assets/pipeline.svg" width="860" alt="카메라 프레임은 고정된 DINOv2 인코더를 거치고, 3,300만 파라미터 생성 모델이 그 토큰과 노이즈를 섞은 직전 묶음을 받아 8프레임 잠재 묶음을 내며, TAESD가 이를 화면으로 바꿉니다."></p>

| 머스크의 글 | RealView |
|---|---|
| 테슬라 범용 비전 인코더(UVE) 특징값 | 고정된 DINOv2-small(Apache-2.0). 16×12 패치 토큰을 8×6 = 프레임당 48토큰으로 |
| "UVE 특징값에서 갈라져 나옴" | 인코더 토큰을 두 번 넣음: 같은 위치의 8×6 잠재 패치에 더하고, in-context 토큰으로도 |
| "별도의 ~10억 파라미터 확산 모델" | 3,300만 파라미터 확산 트랜스포머(12블록, 폭 384). 같은 모양, 30분의 1 크기 |
| "조건부 mean-flow 모델" | Mean-flow 목표([Geng et al. 2025](https://arxiv.org/abs/2505.13447)), 묶음당 계산 한 번 |
| "직전 생성 화면에 노이즈를 더해 디노이즈" | 흐름의 출발점이 순수 노이즈가 아니라 직전 묶음 + 가우시안 노이즈(σ = 0.7). t = 1에서 0까지 한 번에 |
| UVE 18 Hz마다 "8장을 36 fps로" | 인코더 10 Hz 4프레임에서 20 fps 8프레임. 같은 2:1 비율을 comma 카메라 20 fps에 맞춤 |
| "UI 화면 해상도" | TAESD 오토인코더로 256×192 RGB(4×24×32 잠재) |

글에 없어서 우리가 정한 세 가지:

- **표류.** 자기 출력을 먹고 사는 모델은 점점 엇나갑니다. 학습 배치의 4분의 1은 "직전 묶음" 자리에 모델이 직접 그린 것을 넣고, 30%는 아무것도 없이 시작합니다.
- **위치 고정.** in-context 토큰만으로는 부족했습니다. 초기 학습에서 모델이 특징값을 무시하고 직전 화면만 베꼈습니다. 각 인코더 토큰을 같은 위치 패치에 더하자 해결됐습니다.
- **시간 임베딩.** Mean-flow는 t에 대해 네트워크를 미분합니다. DiT 관례인 `t × 1000`은 이 미분을 폭주시켜 첫 학습이 2,000 스텝에서 발산했습니다. t를 그대로 넣습니다.

## 측정 결과

공개한 `weights/realview.pt`(step 10,000, EMA)로, 학습에서 뺀 두 주행(낮 5클립, 밤 3클립, 각 59.6초)을 빈 화면에서부터 자기회귀로 돌려 쟀습니다.
`scripts/eval_rollout.py`, `scripts/val.py`로 재현할 수 있습니다. 생성한 0.4초 묶음마다 같은 클립에서 가장 가까운 실제 묶음을 찾아, 맞는 순간에서 0.4초 안인지 봅니다.

| 생성 모델 입력 | 같은 순간을 맞힌 비율 | PSNR |
|---|---|---|
| 그 주행의 특징값 | **63.8%** (낮 86.2%, 밤 26.4%) | **24.1 dB** |
| 다른 주행의 특징값 | 1.5% | 14.9 dB |
| 특징값 0 | 1.7% | 16.6 dB |
| 무작위 | 2.0% | |

M4 Max에서 8프레임 묶음당 92.6 ms(인코더·생성·디코더 포함, 학습과 GPU를 함께 쓰던 상태), 초당 86프레임입니다.
학습은 배치 16으로 1만 스텝, 같은 맥북에서 1시간 46분. 체크포인트와 노이즈 크기는 같은 시험 주행을 보며 골랐습니다.
나머지 수치와 학습 방법은 [영문 README](README.md#numbers)에 있습니다.

## 한계

- 차 한 대, 카메라 하나, 2018년 베이 에어리어 고속도로. 시내 도로와 다른 카메라에서는 더 나쁠 수 있습니다.
- 256×192라 작은 글씨, 먼 차선, 먼 차가 뭉개집니다.
- 앞차가 자주 사라지거나 뭉개지고, 표지판은 색 덩어리가 됩니다. 운전 판단에 쓸 수 없습니다.
- 밤은 낮보다 훨씬 약합니다(26% 대 86%).
- 노트북 한 대 몇 시간, 3,300만 파라미터입니다. 제품이 아니라 구조를 충실히 작게 재현한 것으로 봐 주세요.

## 라이선스

코드와 가중치는 Apache-2.0, © 2026 AZET LLC. 학습 데이터와 샘플 영상은 comma2k19(MIT, © comma.ai).
DINOv2(Apache-2.0, Meta), TAESD(MIT, © Ollin Boer Bohan). 질문과 내 블랙박스 실행 결과는
[Discussions](https://github.com/azet-llc/realview/discussions)에 올려 주세요.
