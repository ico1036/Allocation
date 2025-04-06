# 포트폴리오 백테스팅 시스템

이 프로젝트는 다양한 포트폴리오 전략을 백테스팅하고 성과를 비교할 수 있는 시스템입니다. 특히 Risk Parity 방식의 포트폴리오 최적화를 지원하며, 여러 포트폴리오 구성의 성과를 시각적으로 비교할 수 있습니다.

## 주요 기능

- **데이터 로딩**: Yahoo Finance API를 통한 주식 데이터 다운로드
- **포트폴리오 백테스팅**: 다양한 비중 구성에 대한 백테스팅 수행
- **Risk Parity 최적화**: 위험 기여도를 균등하게 분배하는 포트폴리오 비중 최적화
- **성과 분석**: 누적 수익률, 연간 수익률, 변동성, 샤프 비율 등 다양한 성과 지표 계산
- **시각화**: Plotly를 활용한 인터랙티브 그래프와 테이블

## 파일 구성

- **run.py**: 메인 실행 파일로, 백테스팅 로직과 포트폴리오 구성 정의
- **dataloader.py**: Yahoo Finance API를 통한 주식 데이터 다운로드 기능
- **util.py**: 성과 지표 계산, 시각화 등 유틸리티 함수 모음
- **model/**: 모델 관련 파일 디렉토리
- **data/**: 데이터 저장 디렉토리

## 사용 방법

1. 필요한 패키지 설치:
   ```
   pip install pandas numpy yfinance plotly scipy
   ```

2. `run.py` 실행:
   ```
   python run.py
   ```

3. 기본적으로 다음 종목들에 대한 백테스팅이 수행됩니다:
   - PLTR (Palantir Technologies)
   - NVDA (NVIDIA)
   - COIN (Coinbase)
   - SCHD (Schwab US Dividend Equity ETF)

   > **참고**: `run.py` 파일의 `symbols` 변수를 수정하여 원하는 종목으로 변경할 수 있습니다. Yahoo Finance에서 지원하는 모든 종목 코드를 사용할 수 있습니다.

4. 두 가지 포트폴리오 전략이 비교됩니다:
   - 수동 비중 (각 종목 25%)
   - Risk Parity 최적화 비중

## 주요 함수

### `risk_parity_weights(returns, target_risk_contribution=None, lookback_period=None)`
- Risk Parity 방식으로 포트폴리오 비중을 최적화합니다.
- 각 자산이 포트폴리오 위험에 동일하게 기여하도록 비중을 조정합니다.
- 과거 기간을 지정하여 비중 계산에 사용할 데이터 기간을 제한할 수 있습니다.

### `light_backtest(stock_data, weights, start_date=None, end_date=None)`
- 포트폴리오 백테스트를 수행하고 결과를 시각화합니다.
- 시작일과 종료일을 지정하여 백테스트 기간을 제한할 수 있습니다.

### `calculate_performance_metrics(returns)`
- 포트폴리오 수익률에 대한 성과 지표를 계산합니다.
- 연간 수익률, 연간 변동성, 샤프 비율 등을 계산합니다.

### `plot_portfolio_return_comparison(portfolio_returns_df, comparison_df)`
- 여러 포트폴리오의 누적 수익률을 비교하는 그래프와 성과 비교 테이블을 생성합니다.

## 성과 지표

- **누적 수익률**: 전체 기간 동안의 총 수익률
- **연간 수익률**: 연평균 복합 성장률 (CAGR)
- **연간 변동성**: 일간 변동성의 연율화 값
- **샤프 비율**: (연간 수익률 - 무위험 수익률) / 연간 변동성

## 참고 사항

- 이 시스템은 교육 및 연구 목적으로 개발되었습니다.
- 실제 투자에 사용하기 전에 충분한 검증이 필요합니다.
- 과거 성과가 미래 수익을 보장하지 않습니다.
