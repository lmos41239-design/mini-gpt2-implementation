# Mini GPT-2 Implementation

GPT-2의 기본 구조를 작게 구현하고, tiny Shakespeare가 아닌 다른 데이터셋으로 학습해 본 프로젝트입니다.

## Dataset

이 프로젝트에서는 tiny Shakespeare를 사용하지 않았습니다.
대신 public domain 문학 텍스트인 `Alice in Wonderland` 일부 문장을 사용했습니다.

데이터셋 파일:

```text
data/alice_wonderland_sample.txt
```

## Model

구현한 모델은 작은 character-level GPT 모델입니다.

주요 구성은 다음과 같습니다.

- token embedding
- positional embedding
- multi-head self-attention
- feed-forward layer
- transformer block
- text generation function

수업 프로젝트에서 구조를 이해하는 것이 목적이기 때문에 모델 크기는 작게 설정했습니다.

## Training result

훈련 결과는 아래 파일에 정리했습니다.

```text
results/training_result.txt
```

간단한 결과:

```text
step 0 train loss: 4.214
step 100 train loss: 2.971
step 300 train loss: 2.341
step 500 train loss: 2.087
```

loss가 처음보다 낮아졌기 때문에, 작은 데이터셋 안에서 문자의 패턴을 어느 정도 학습했다고 볼 수 있습니다.

## Generated sample

생성 결과는 아래 파일에 저장했습니다.

```text
samples/generated_sample.txt
```

## How to run

```bash
pip install -r requirements.txt
python train.py
python generate.py
```

## Note

이 프로젝트는 큰 GPT-2 모델을 그대로 재현한 것이 아니라,
GPT 방식의 핵심 구조를 작게 구현한 수업용 프로젝트입니다.
