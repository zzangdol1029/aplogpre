# 로그 이상치 탐지 시스템

Spring Boot 로그 파일을 분석하여 이상 패턴을 탐지하는 시스템입니다.

## 📋 기능

1. **로그 파싱**: Spring Boot 로그 형식 자동 파싱
2. **특징 추출**: 시간 윈도우별 통계 특징 추출
3. **다양한 이상치 탐지 방법**:
   - 통계적 이상치 탐지 (Z-score 기반)
   - 에러 급증 탐지
   - 비정상 패턴 탐지 (에러 집중, 로그 빈도 이상, 새로운 예외 타입)
   - 머신러닝 기반 탐지 (Isolation Forest, AutoEncoder)

## 🚀 사용 방법

### 1단계: 모델 학습

먼저 기존 로그 데이터로 모델을 학습합니다:

```bash
cd /Users/zzangdol/PycharmProjects/zzangdol/pattern/prelog
python log_anomaly_detector.py
```

이 명령어는:
- `logs/backup` 폴더의 로그 파일을 파싱
- 특징을 추출하고 기준선 통계 계산
- 머신러닝 모델 학습
- 학습된 모델을 `results/trained_model.pkl`에 저장

### 2단계: 새로운 로그 데이터로 이상치 탐지

#### 방법 1: 명령어로 실행

```bash
python log_anomaly_detector.py test
```

#### 방법 2: Python 스크립트로 실행

```python
from log_anomaly_detector import LogAnomalyDetectionSystem
import os

# 학습된 모델 경로
model_path = "results/trained_model.pkl"

# 새로운 로그 디렉토리
new_log_dir = "logs/new_logs"  # 새로운 로그 파일이 있는 경로

# 시스템 초기화
system = LogAnomalyDetectionSystem(new_log_dir)

# 모델 로드
system.load_model(model_path)

# 이상치 탐지
results = system.detect_anomalies_on_new_data(
    new_log_dir,
    max_files=None,      # None이면 전체 파일 처리
    sample_lines=None    # None이면 전체 라인 처리
)

# 리포트 생성
system.generate_report(results)
```

#### 방법 3: 테스트 스크립트 사용

```bash
python test_anomaly_detection.py
```

또는 Python에서:

```python
from test_anomaly_detection import detect_anomalies_in_new_logs

# 새로운 로그 디렉토리
new_log_dir = "logs/new_logs"

# 이상치 탐지
results = detect_anomalies_in_new_logs(new_log_dir)
```

## 📊 결과 확인

이상치 탐지 결과는 다음 위치에 저장됩니다:

- `results/anomalies_statistical.csv`: 통계적 이상치
- `results/anomalies_error_spikes.csv`: 에러 급증
- `results/anomalies_unusual_patterns.csv`: 비정상 패턴
- `results/anomalies_ml_isolation_forest.csv`: ML 기반 이상치

## ⚙️ 설정 옵션

### 학습 시 샘플링 옵션

`log_anomaly_detector.py`의 `main()` 함수에서:

```python
MAX_FILES = 5        # 처리할 최대 파일 수 (None이면 전체)
SAMPLE_LINES = 10000 # 파일당 최대 처리할 라인 수 (None이면 전체)
```

### 테스트 시 샘플링 옵션

```python
results = system.detect_anomalies_on_new_data(
    new_log_dir,
    max_files=10,      # 처음 10개 파일만 처리
    sample_lines=5000  # 파일당 5000줄만 처리
)
```

## 🔍 탐지되는 이상 패턴

1. **통계적 이상치**: 평균 대비 3 표준편차 이상 벗어난 패턴
2. **에러 급증**: 기준 에러율 대비 5배 이상 증가
3. **에러 집중**: 특정 클래스에서 전체 에러의 50% 이상 발생
4. **로그 빈도 이상**: 평소 대비 로그 빈도가 급격히 증가/감소
5. **새로운 예외 타입**: 기존에 없던 새로운 예외 발생

## 📝 예시 출력

```
============================================================
새로운 로그 데이터 이상치 탐지
============================================================

새로운 로그 파일 파싱 중...
✅ 1500개 로그 라인 파싱 완료

특징 추출 중...
✅ 5개 시간 윈도우 특징 추출 완료

이상치 탐지 중...
   ✅ 통계적 이상치: 1개
   ✅ 에러 급증: 2개
   ✅ 비정상 패턴: 0개
   ✅ ML 이상치 (IF): 1개

============================================================
이상치 탐지 결과 리포트
============================================================

🚨 에러 급증:
   시간: 2025-07-10 14:30:00
   기준 에러율: 2.60%
   현재 에러율: 15.45%
   배수: 5.9배
   에러 수: 3개 / 총 1016개
```

## 💡 팁

- **전체 데이터로 학습**: 더 정확한 모델을 위해 `MAX_FILES = None`, `SAMPLE_LINES = None`으로 설정
- **실시간 모니터링**: 새로운 로그 파일이 생성될 때마다 `detect_anomalies_on_new_data()` 호출
- **결과 분석**: CSV 파일을 Excel이나 pandas로 열어서 상세 분석 가능

