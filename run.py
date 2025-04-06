#!/usr/bin/env .venv/bin/python3.10
from dataloader import get_stock_data
import pandas as pd
import numpy as np
from util import calculate_performance_metrics, visualize_backtest_results, plot_portfolio_return_comparison
from scipy.optimize import minimize



def risk_parity_weights(returns, target_risk_contribution=None, lookback_period=None):
    """
    Risk Parity 방식으로 포트폴리오 비중을 최적화합니다.
    
    Parameters:
    -----------
    returns : pandas.DataFrame
        종목별 일별 수익률 데이터
    target_risk_contribution : list, optional
        각 자산의 목표 위험 기여도 (기본값: 균등 분배)
    lookback_period : int, optional
        비중 계산에 사용할 과거 기간 (일 단위, 기본값: 전체 기간)
    
    Returns:
    --------
    dict
        최적화된 종목별 비중
    """
    # 과거 기간 설정
    if lookback_period is not None:
        returns = returns.iloc[-lookback_period:]
    
    # 공분산 행렬 계산
    cov_matrix = returns.cov() * 252  # 연율화 공분산 행렬
    
    # 목표 위험 기여도 설정 (기본값: 균등 분배)
    n_assets = len(returns.columns)
    if target_risk_contribution is None:
        target_risk_contribution = [1/n_assets] * n_assets
    
    # 위험 기여도 계산 함수
    def risk_contribution(weights):
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        marginal_risk_contribution = np.dot(cov_matrix, weights) / portfolio_vol
        risk_contribution = weights * marginal_risk_contribution
        return risk_contribution
    
    # 위험 기여도 차이의 제곱합을 최소화하는 목적 함수
    def objective(weights):
        risk_contrib = risk_contribution(weights)
        target_risk = np.array(target_risk_contribution) * np.sum(risk_contrib)
        return np.sum((risk_contrib - target_risk) ** 2)
    
    # 제약 조건: 비중 합계 = 1, 비중 >= 0
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 비중 합계 = 1
    ]
    bounds = tuple((0, 1) for _ in range(n_assets))  # 비중 >= 0
    
    # 초기 비중 설정 (균등 분배)
    initial_weights = np.array([1/n_assets] * n_assets)
    
    # 최적화 수행
    result = minimize(
        objective,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # 최적화된 비중을 딕셔너리로 변환
    optimized_weights = {symbol: weight for symbol, weight in zip(returns.columns, result.x)}
    
    return optimized_weights


def light_backtest(stock_data, weights, start_date=None, end_date=None):
    """
    포트폴리오 백테스트를 수행하고 결과를 시각화합니다.
    
    Parameters:
    -----------
    stock_data : pandas.DataFrame
        종목별 주가 데이터 (날짜 인덱스, 종목 컬럼)
    weights : dict
        종목별 비중 (예: {'AAPL': 0.3, 'MSFT': 0.7})
    start_date : str, optional
        백테스트 시작일 (YYMMDD 형식)
    end_date : str, optional
        백테스트 종료일 (YYMMDD 형식)
    
    Returns:
    --------
    tuple
        (포트폴리오 일별 수익률, 성과 지표 DataFrame)
    """
    # 비중 합계가 1인지 확인
    if abs(sum(weights.values()) - 1.0) > 0.0001:
        raise ValueError("비중의 합이 1이어야 합니다.")
    
    # 데이터 기간 필터링
    if start_date:
        stock_data = stock_data[stock_data.index >= start_date]
    if end_date:
        stock_data = stock_data[stock_data.index <= end_date]
    
    # 비중을 numpy 배열로 변환 (1 x n_assets)
    weight_array = np.array([weights[symbol] for symbol in stock_data.columns]).reshape(1, -1)
    
    # 일별 수익률 계산
    returns = stock_data.pct_change()
    
    # 첫날의 수익률을 0으로 설정
    returns.iloc[0] = 0

    # 포트폴리오 수익률 계산 (행렬 곱셈)
    portfolio_returns = returns.dot(weight_array.T)
    
    # 누적 수익률 계산 (cumprod 사용)
    # 첫날부터 시작하여 매일의 수익률을 누적
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    
    # 성과 지표 계산
    performance_df = calculate_performance_metrics(portfolio_returns)
    
    # 시각화 함수 호출
    visualize_backtest_results(cumulative_returns, returns, weights, stock_data, performance_df)
    
    print(f"portfolio_returns 형태: {portfolio_returns.shape}")
    
    return portfolio_returns, performance_df






if __name__ == "__main__":
    symbols = ['PLTR', 'NVDA', 'COIN', 'SCHD']
    stock_data = get_stock_data(symbols)
    
    # 일별 수익률 계산
    returns = stock_data.pct_change()
    returns.iloc[0] = 0  # 첫날의 수익률을 0으로 설정
    
    # Risk Parity 방식으로 비중 최적화 (과거 6개월 데이터 사용)
    lookback_period = 126  # 약 6개월 (거래일 기준)
    risk_parity_weights_dict = risk_parity_weights(returns, lookback_period=lookback_period)
    print(f"\n=== Risk Parity 최적화 비중 (과거 {lookback_period}일 기준) ===")
    for symbol, weight in risk_parity_weights_dict.items():
        print(f"{symbol}: {weight:.4f}")
    
    # 기존 비중 설정
    manual_weights = {
        'PLTR': 0.25,
        'NVDA': 0.25,
        'COIN': 0.25,
        'SCHD': 0.25
    }
    
    # 여러 포트폴리오 비중 설정
    portfolio_weights = {
        '수동 비중': manual_weights,
        'Risk Parity': risk_parity_weights_dict
    }
    
    # 각 포트폴리오에 대한 백테스트 실행 및 결과 저장
    results = {}
    portfolio_returns_dict = {}  # 포트폴리오별 수익률 저장
    
    for portfolio_name, weights in portfolio_weights.items():
        print(f"\n\n=== {portfolio_name} 백테스트 결과 ===")
        returns, performance_df = light_backtest(stock_data, weights)
        
        # 수익률 통계 출력
        print(f"\n=== {portfolio_name} 수익률 통계 ===")
        print(performance_df)
        
        # 결과 저장
        results[portfolio_name] = {
            'returns': returns,
            'performance': performance_df.loc['Total Period']
        }
        
        # 포트폴리오 수익률 저장
        portfolio_returns_dict[portfolio_name] = returns.values.flatten()
    
    # 포트폴리오별 수익률을 날짜를 인덱스로 하는 DataFrame으로 변환
    portfolio_returns_df = pd.DataFrame(
        portfolio_returns_dict,
        index=stock_data.index  # 전체 인덱스 사용
    )
    
    # 포트폴리오 비교
    print("\n\n=== 포트폴리오 비교 ===")
    comparison_data = {
        '포트폴리오': [],
        '누적 수익률': [],
        '연간 수익률': [],
        '연간 변동성': [],
        '샤프 비율': []
    }
    
    for portfolio_name, result in results.items():
        comparison_data['포트폴리오'].append(portfolio_name)
        comparison_data['누적 수익률'].append((1 + result['returns'].values).prod() - 1)
        comparison_data['연간 수익률'].append(result['performance']['Annual Return'])
        comparison_data['연간 변동성'].append(result['performance']['Annual Volatility'])
        comparison_data['샤프 비율'].append(result['performance']['Sharpe Ratio'])
    
    # 비교 결과를 DataFrame으로 변환
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.set_index('포트폴리오', inplace=True)
    
    # 결과 출력
    print("\n=== 포트폴리오 성과 비교 ===")
    print(comparison_df)
    
    # 포트폴리오별 수익률 테이블 출력
    print("\n=== 포트폴리오별 일별 수익률 (처음 5일) ===")
    print(portfolio_returns_df.head())
    
    # 포트폴리오별 누적 수익률 비교 그래프와 성과 비교 테이블 생성
    print("\n=== 포트폴리오 성과 분석 ===")
    fig = plot_portfolio_return_comparison(portfolio_returns_df, comparison_df)
    
