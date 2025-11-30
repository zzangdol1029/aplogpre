"""
새로운 로그 데이터에 대해 이상치 탐지하는 예시 스크립트
"""

from log_anomaly_detector import LogAnomalyDetectionSystem
import os


def detect_anomalies_in_new_logs(new_log_directory, model_path=None):
    """
    새로운 로그 디렉토리에서 이상치를 탐지합니다.
    
    Args:
        new_log_directory: 새로운 로그 파일이 있는 디렉토리 경로
        model_path: 학습된 모델 경로 (None이면 기본 경로 사용)
    
    Returns:
        dict: 이상치 탐지 결과
    """
    # 기본 모델 경로
    if model_path is None:
        model_path = "/Users/zzangdol/PycharmProjects/zzangdol/pattern/prelog/results/trained_model.pkl"
    
    # 모델이 없으면 에러
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"학습된 모델을 찾을 수 없습니다: {model_path}\n"
            "먼저 log_anomaly_detector.py를 실행하여 모델을 학습하세요."
        )
    
    # 시스템 초기화
    system = LogAnomalyDetectionSystem(new_log_directory)
    
    # 모델 로드
    print("=" * 60)
    print("학습된 모델 로드 중...")
    print("=" * 60)
    system.load_model(model_path)
    
    # 새로운 로그 데이터로 이상치 탐지
    results = system.detect_anomalies_on_new_data(
        new_log_directory,
        max_files=None,  # None이면 전체 파일 처리
        sample_lines=None  # None이면 전체 라인 처리
    )
    
    # 리포트 생성
    system.generate_report(results)
    
    return results


def detect_anomalies_in_single_file(log_file_path, model_path=None):
    """
    단일 로그 파일에 대해 이상치를 탐지합니다.
    
    Args:
        log_file_path: 로그 파일 경로
        model_path: 학습된 모델 경로
    
    Returns:
        dict: 이상치 탐지 결과
    """
    # 파일이 있는 디렉토리로 변환
    log_directory = os.path.dirname(log_file_path)
    
    # 파일명으로 필터링하기 위해 임시 디렉토리 생성
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, os.path.basename(log_file_path))
    shutil.copy2(log_file_path, temp_file)
    
    try:
        results = detect_anomalies_in_new_logs(temp_dir, model_path)
    finally:
        shutil.rmtree(temp_dir)
    
    return results


if __name__ == "__main__":
    # 예시 1: 새로운 로그 디렉토리 전체 분석
    new_log_dir = "/Users/zzangdol/PycharmProjects/zzangdol/pattern/prelog/logs/backup"
    
    print("=" * 60)
    print("새로운 로그 데이터 이상치 탐지")
    print("=" * 60)
    
    try:
        results = detect_anomalies_in_new_logs(new_log_dir)
        
        # 결과 저장
        output_dir = "/Users/zzangdol/PycharmProjects/zzangdol/pattern/prelog/results"
        os.makedirs(output_dir, exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, df in results.items():
            if not df.empty:
                output_path = os.path.join(output_dir, f"new_anomalies_{name}_{timestamp}.csv")
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"\n💾 결과 저장: {output_path}")
    
    except FileNotFoundError as e:
        print(f"\n❌ 오류: {e}")
        print("\n먼저 다음 명령어로 모델을 학습하세요:")
        print("  python log_anomaly_detector.py")
    
    # 예시 2: 단일 파일 분석
    # single_file = "/path/to/single/log/file.log"
    # results = detect_anomalies_in_single_file(single_file)

