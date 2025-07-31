import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
plt.rcParams['font.family'] = ['Malgun Gothic']
plt.rcParams['axes.unicode_minus'] = False 
import seaborn as sns 
from datetime import datetime, timedelta
import random 
from typing import List, Dict, Tuple
import json
import os 


class RFWakeupDataGenerator:
    def __init__(self, seed: int = 42):
        """
        RF Wake-up 이벤트 로그 데이터 생성기
        
        Args:
            seed: 랜덤 시드
        """
        np.random.seed(seed)
        random.seed(seed)
        
        # 기본 설정값들
        self.normal_rssi_range = (-70, -30)  # dBm
        self.anomaly_rssi_range = (-100, -20)  # 더 넓은 범위
        self.frequency_channels = [2.4, 5.0, 6.0]  # GHz
        self.device_types = ['sensor', 'actuator', 'gateway', 'mobile']
        self.locations = ['office', 'warehouse', 'lab', 'outdoor']
        
    def generate_timestamp_sequence(self, 
                                  start_time: datetime, 
                                  num_events: int,
                                  duration_hours: int = 168,  # 1주일
                                  base_interval_minutes: int = 15,
                                  jitter_minutes: int = 5) -> List[datetime]:
        """
        타임스탬프 시퀀스 생성
        
        Args:
            start_time: 시작 시간
            num_events: 생성할 이벤트 수
            duration_hours: 지속 시간 (시간) - 기본 1주일
            base_interval_minutes: 기본 간격 (분)
            jitter_minutes: 시간 변동 범위 (분)
        """
        timestamps = []
        
        # 요청된 이벤트 수를 보장하기 위해 적응적 간격 계산
        if num_events > 0:
            # 전체 시간을 이벤트 수로 나누어 평균 간격 계산
            avg_interval_minutes = (duration_hours * 60) / num_events
            # 기본 간격이 너무 크면 조정
            if base_interval_minutes > avg_interval_minutes:
                base_interval_minutes = max(1, int(avg_interval_minutes * 0.8))
        
        current_time = start_time
        end_time = start_time + timedelta(hours=duration_hours)
        
        # 먼저 균등하게 분포된 시간점들을 생성
        for i in range(num_events):
            if len(timestamps) >= num_events:
                break
                
            # 균등 분포 + 지터
            base_offset_minutes = (i * duration_hours * 60) / num_events
            jitter = np.random.randint(-jitter_minutes, jitter_minutes + 1) if jitter_minutes > 0 else 0
            offset_minutes = base_offset_minutes + jitter
            
            timestamp = start_time + timedelta(minutes=offset_minutes)
            
            # 시간 범위 내에 있는지 확인
            if timestamp <= end_time:
                timestamps.append(timestamp)
        
        # 부족한 경우 추가 생성
        while len(timestamps) < num_events:
            # 랜덤한 시간에 추가 이벤트 생성
            random_offset = np.random.uniform(0, duration_hours * 60)
            additional_timestamp = start_time + timedelta(minutes=random_offset)
            timestamps.append(additional_timestamp)
        
        # 시간순 정렬 후 중복 제거
        timestamps = sorted(list(set(timestamps)))
        
        # 정확히 요청된 수만큼 반환
        return timestamps[:num_events]
    
    def generate_normal_events(self, 
                             num_events: int,
                             start_time: datetime = None) -> pd.DataFrame:
        """
        정상적인 RF wake-up 이벤트 생성
        
        Args:
            num_events: 생성할 이벤트 수
            start_time: 시작 시간
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(days=7)
        
        events = []
        
        # 타임스탬프 생성 - 정확한 개수 보장
        timestamps = self.generate_timestamp_sequence(
            start_time, 
            num_events=num_events,
            duration_hours=168,  # 1주일
            base_interval_minutes=15,  # 더 짧은 간격으로 조정
            jitter_minutes=5
        )
        
        print(f"생성된 타임스탬프 수: {len(timestamps)}, 요청 수: {num_events}")
        
        for i, timestamp in enumerate(timestamps):
            # 정상적인 패턴의 특성
            base_rssi = np.random.normal(-50, 8)  # 평균 -50dBm, 표준편차 8
            rssi = np.clip(base_rssi, self.normal_rssi_range[0], self.normal_rssi_range[1])
            
            # 시간대에 따른 패턴 (업무시간에 더 많은 활동)
            hour = timestamp.hour
            if 9 <= hour <= 18:  # 업무시간
                activity_factor = np.random.normal(1.2, 0.2)
            else:  # 비업무시간
                activity_factor = np.random.normal(0.8, 0.3)
            
            event = {
                'timestamp': timestamp,
                'device_id': f'DEV_{i % 50:03d}',  # 더 많은 디바이스 사용
                'device_type': np.random.choice(self.device_types),
                'location': np.random.choice(self.locations),
                'rssi_dbm': round(rssi, 1),
                'frequency_ghz': np.random.choice(self.frequency_channels),
                'channel': np.random.randint(1, 12),
                'signal_duration_ms': np.random.normal(50, 10),  # 평균 50ms
                'response_time_ms': np.random.normal(5, 1),  # 평균 5ms 응답시간
                'packet_loss_rate': np.random.exponential(0.01),  # 낮은 패킷 손실률
                'retry_count': np.random.poisson(0.1),  # 낮은 재시도 횟수
                'battery_level': np.random.normal(80, 15),  # 배터리 레벨
                'temperature_c': np.random.normal(25, 5),  # 온도
                'humidity_percent': np.random.normal(45, 10),  # 습도
                'interference_level': np.random.exponential(2),  # 간섭 레벨
                'label': 'normal'
            }
            
            # 값 범위 조정
            event['signal_duration_ms'] = max(10, event['signal_duration_ms'])
            event['response_time_ms'] = max(1, event['response_time_ms'])
            event['packet_loss_rate'] = min(1.0, max(0, event['packet_loss_rate']))
            event['battery_level'] = min(100, max(0, event['battery_level']))
            event['temperature_c'] = min(60, max(-10, event['temperature_c']))
            event['humidity_percent'] = min(100, max(0, event['humidity_percent']))
            
            events.append(event)
        
        result_df = pd.DataFrame(events)
        print(f"생성된 정상 이벤트 수: {len(result_df)}")
        return result_df
    
    def generate_anomaly_events(self, 
                              num_events: int,
                              anomaly_types: List[str] = None) -> pd.DataFrame:
        """
        이상적인 RF wake-up 이벤트 생성
        
        Args:
            num_events: 생성할 이벤트 수
            anomaly_types: 이상 유형 리스트
        """
        if anomaly_types is None:
            anomaly_types = ['signal_anomaly', 'timing_anomaly', 'frequency_anomaly', 
                           'interference_anomaly', 'hardware_anomaly']
        
        events = []
        start_time = datetime.now() - timedelta(days=7)
        
        for i in range(num_events):
            anomaly_type = np.random.choice(anomaly_types)
            
            # 기본 시간 (더 랜덤하게)
            random_offset = np.random.randint(0, 168 * 60)  # 1주일 내 랜덤 분
            timestamp = start_time + timedelta(minutes=random_offset)
            
            # 기본 이벤트 구조
            event = {
                'timestamp': timestamp,
                'device_id': f'DEV_{np.random.randint(0, 30):03d}',  # 더 많은 디바이스 포함
                'device_type': np.random.choice(self.device_types),
                'location': np.random.choice(self.locations),
                'anomaly_type': anomaly_type,
                'label': 'anomaly'
            }
            
            # 이상 유형별 특성 생성
            if anomaly_type == 'signal_anomaly':
                # 비정상적인 신호 강도
                event['rssi_dbm'] = np.random.choice([
                    np.random.normal(-90, 5),  # 매우 약한 신호
                    np.random.normal(-20, 3),  # 매우 강한 신호
                ])
                event['signal_duration_ms'] = np.random.choice([
                    np.random.normal(200, 50),  # 비정상적으로 긴 신호
                    np.random.normal(5, 2)      # 비정상적으로 짧은 신호
                ])
                
            elif anomaly_type == 'timing_anomaly':
                # 정상 신호 강도이지만 비정상적인 타이밍
                event['rssi_dbm'] = np.random.normal(-50, 8)
                event['response_time_ms'] = np.random.exponential(50)  # 긴 응답시간
                event['signal_duration_ms'] = np.random.normal(50, 10)
                
            elif anomaly_type == 'frequency_anomaly':
                # 비정상적인 주파수 사용
                event['rssi_dbm'] = np.random.normal(-50, 8)
                event['frequency_ghz'] = np.random.choice([1.8, 3.5, 7.2])  # 비표준 주파수
                event['channel'] = np.random.randint(50, 100)  # 비정상적인 채널
                event['signal_duration_ms'] = np.random.normal(50, 10)
                
            elif anomaly_type == 'interference_anomaly':
                # 높은 간섭으로 인한 이상
                event['rssi_dbm'] = np.random.normal(-60, 15)
                event['interference_level'] = np.random.exponential(20)  # 높은 간섭
                event['packet_loss_rate'] = np.random.beta(2, 3)  # 높은 패킷 손실
                event['retry_count'] = np.random.poisson(5)  # 많은 재시도
                event['signal_duration_ms'] = np.random.normal(80, 20)
                
            elif anomaly_type == 'hardware_anomaly':
                # 하드웨어 문제로 인한 이상
                event['rssi_dbm'] = np.random.normal(-55, 20)
                event['battery_level'] = np.random.exponential(10)  # 낮은 배터리
                event['temperature_c'] = np.random.choice([
                    np.random.normal(60, 5),   # 과열
                    np.random.normal(-5, 3)    # 저온
                ])
                event['signal_duration_ms'] = np.random.normal(50, 10)
            
            # 공통 필드 설정 (누락된 경우)
            defaults = {
                'frequency_ghz': np.random.choice(self.frequency_channels),
                'channel': np.random.randint(1, 12),
                'response_time_ms': np.random.normal(5, 1),
                'packet_loss_rate': np.random.exponential(0.01),
                'retry_count': np.random.poisson(0.1),
                'battery_level': np.random.normal(80, 15),
                'temperature_c': np.random.normal(25, 5),
                'humidity_percent': np.random.normal(45, 10),
                'interference_level': np.random.exponential(2),
                'signal_duration_ms': np.random.normal(50, 10)
            }
            
            for key, default_value in defaults.items():
                if key not in event:
                    event[key] = default_value
            
            # 값 범위 조정
            event['rssi_dbm'] = round(event['rssi_dbm'], 1)
            event['signal_duration_ms'] = max(1, event['signal_duration_ms'])
            event['response_time_ms'] = max(0.1, event['response_time_ms'])
            event['packet_loss_rate'] = min(1.0, max(0, event['packet_loss_rate']))
            event['battery_level'] = min(100, max(0, event['battery_level']))
            event['temperature_c'] = min(80, max(-20, event['temperature_c']))
            event['humidity_percent'] = min(100, max(0, event['humidity_percent']))
            
            events.append(event)
        
        return pd.DataFrame(events)
    
    def generate_dataset(self, 
                        normal_ratio: float = 0.8,
                        total_samples: int = 10000,
                        save_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        훈련용과 테스트용 데이터셋 생성
        
        Args:
            normal_ratio: 정상 데이터 비율
            total_samples: 전체 샘플 수
            save_path: 저장 경로
        """
        normal_samples = int(total_samples * normal_ratio)
        anomaly_samples = total_samples - normal_samples
        
        print(f"정상 샘플 생성 중: {normal_samples}개")
        normal_data = self.generate_normal_events(normal_samples)
        
        print(f"이상 샘플 생성 중: {anomaly_samples}개")
        anomaly_data = self.generate_anomaly_events(anomaly_samples)
        
        # 전체 데이터 결합
        full_data = pd.concat([normal_data, anomaly_data], ignore_index=True)
        
        # 시간순 정렬
        full_data = full_data.sort_values('timestamp').reset_index(drop=True)
        
        # 훈련/테스트 분할 (시계열 특성 고려)
        split_idx = int(len(full_data) * 0.7)
        train_data = full_data[:split_idx].copy()
        test_data = full_data[split_idx:].copy()
        
        os.makedirs(save_path, exist_ok=True)
        
        if save_path:
            train_data.to_csv(rf"{save_path}\train.csv", index=False)
            test_data.to_csv(rf"{save_path}\test.csv", index=False)
            
            # 메타데이터 저장
            metadata = {
                'generation_time': datetime.now().isoformat(),
                'total_samples': total_samples,
                'normal_ratio': normal_ratio,
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'features': list(full_data.columns),
                'anomaly_types': list(anomaly_data['anomaly_type'].unique()) if 'anomaly_type' in anomaly_data.columns else []
            }
            
            with open(rf"{save_path}\metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        print(f"\n데이터셋 생성 완료!")
        print(f"전체: {len(full_data)}개 (정상: {len(full_data[full_data['label']=='normal'])}개, "
              f"이상: {len(full_data[full_data['label']=='anomaly'])}개)")
        print(f"훈련: {len(train_data)}개, 테스트: {len(test_data)}개")
        
        return train_data, test_data
    
    def visualize_data(self, data: pd.DataFrame, save_path: str = None):
        """
        생성된 데이터 시각화
        
        Args:
            data: 시각화할 데이터
            save_path: 그래프 저장 경로
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('RF Wake-up Event Data Analysis', fontsize=16)
        
        # 1. RSSI 분포
        axes[0, 0].hist(data[data['label']=='normal']['rssi_dbm'], 
                       alpha=0.7, label='Normal', bins=30, color='blue')
        axes[0, 0].hist(data[data['label']=='anomaly']['rssi_dbm'], 
                       alpha=0.7, label='Anomaly', bins=30, color='red')
        axes[0, 0].set_xlabel('RSSI (dBm)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('RSSI Distribution')
        axes[0, 0].legend()
        
        # 2. 시간별 이벤트 발생
        data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
        hourly_normal = data[data['label']=='normal']['hour'].value_counts().sort_index()
        hourly_anomaly = data[data['label']=='anomaly']['hour'].value_counts().sort_index()
        
        axes[0, 1].bar(hourly_normal.index, hourly_normal.values, 
                      alpha=0.7, label='Normal', color='blue')
        axes[0, 1].bar(hourly_anomaly.index, hourly_anomaly.values, 
                      alpha=0.7, label='Anomaly', color='red')
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_ylabel('Event Count')
        axes[0, 1].set_title('Events by Hour')
        axes[0, 1].legend()
        
        # 3. 신호 지속시간 vs 응답시간
        normal_data = data[data['label']=='normal']
        anomaly_data = data[data['label']=='anomaly']
        
        axes[0, 2].scatter(normal_data['signal_duration_ms'], normal_data['response_time_ms'],
                          alpha=0.6, label='Normal', color='blue', s=20)
        axes[0, 2].scatter(anomaly_data['signal_duration_ms'], anomaly_data['response_time_ms'],
                          alpha=0.6, label='Anomaly', color='red', s=20)
        axes[0, 2].set_xlabel('Signal Duration (ms)')
        axes[0, 2].set_ylabel('Response Time (ms)')
        axes[0, 2].set_title('Signal Duration vs Response Time')
        axes[0, 2].legend()
        
        # 4. 패킷 손실률 분포
        axes[1, 0].hist(normal_data['packet_loss_rate'], 
                       alpha=0.7, label='Normal', bins=30, color='blue')
        axes[1, 0].hist(anomaly_data['packet_loss_rate'], 
                       alpha=0.7, label='Anomaly', bins=30, color='red')
        axes[1, 0].set_xlabel('Packet Loss Rate')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Packet Loss Rate Distribution')
        axes[1, 0].legend()
        
        # 5. 온도 vs 배터리 레벨
        axes[1, 1].scatter(normal_data['temperature_c'], normal_data['battery_level'],
                          alpha=0.6, label='Normal', color='blue', s=20)
        axes[1, 1].scatter(anomaly_data['temperature_c'], anomaly_data['battery_level'],
                          alpha=0.6, label='Anomaly', color='red', s=20)
        axes[1, 1].set_xlabel('Temperature (°C)')
        axes[1, 1].set_ylabel('Battery Level (%)')
        axes[1, 1].set_title('Temperature vs Battery Level')
        axes[1, 1].legend()
        
        # 6. 이상 유형별 분포 (이상 데이터만)
        if 'anomaly_type' in anomaly_data.columns:
            anomaly_counts = anomaly_data['anomaly_type'].value_counts()
            axes[1, 2].pie(anomaly_counts.values, labels=anomaly_counts.index, autopct='%1.1f%%')
            axes[1, 2].set_title('Anomaly Types Distribution')
        else:
            axes[1, 2].text(0.5, 0.5, 'No anomaly type data', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(rf"{save_path}\analysis.png", dpi=300, bbox_inches='tight')
        
        plt.show()
