#!/usr/bin/env .venv/bin/python3.10
from dataloader import get_stock_data
import pandas as pd
import numpy as np
from util import (calculate_performance_metrics, visualize_backtest_results, 
                 plot_portfolio_return_comparison, setup_logging, get_rebalance_dates, 
                 handle_rebalancing_warning)
from model import risk_parity_weights, create_weight_calculator
import matplotlib.pyplot as plt
import logging

# 전역 설정 파라미터
CONFIG = {
    # 종목 관련 설정
    'symbols': ['PLTR', 'NVDA', 'SGOV'],
    
    # 최적화 관련 설정
    'optimization': {
        'min_weight': 0.05,  # 최소 비중 (5%)
        'max_weight': 0.60,  # 최대 비중 (60%)
        'lookback_period': 126,  # 약 6개월 (거래일 기준)
    },
    
    # 리밸런싱 관련 설정
    'rebalancing': {
        'frequency': 'yearly',  # 리밸런싱 주기 ('yearly', 'quarterly', 'monthly')
        'quarterly_months': [1, 4, 7, 10],  # 분기별 리밸런싱에 사용되는 월
    },
    
    # 포트폴리오 설정
    'portfolios': {
        'equal_weight': {
            'name': '수동 균등 비중',
            'weight_per_asset': 1/3,  # 각 종목당 동일 비중
        },
        'risk_parity': {
            'name': '제약있는 Risk Parity',
        }
    }
}

def light_backtest(stock_data, weights, start_date=None, end_date=None, rebalance_frequency=None, 
                  weight_calculator=None, lookback_period=None, config=None):
    """
    포트폴리오 백테스트를 수행합니다.
    """
    # 설정값 기본값 처리
    if config is not None:
        if rebalance_frequency is None:
            rebalance_frequency = config['rebalancing']['frequency']
        if lookback_period is None:
            lookback_period = config['optimization']['lookback_period']
    
    # 비중 합계가 1인지 확인
    if abs(sum(weights.values()) - 1.0) > 0.001:
        raise ValueError("비중의 합이 1이어야 합니다.")
    
    # 데이터 기간 필터링
    if start_date:
        stock_data = stock_data[stock_data.index >= start_date]
    if end_date:
        stock_data = stock_data[stock_data.index <= end_date]
    
    # 날짜 인덱스를 datetime으로 변환
    stock_data.index = pd.to_datetime(stock_data.index, format='%y%m%d')
    
    # 일별 수익률 계산
    returns = stock_data.pct_change()
    
    # 첫날의 수익률을 0으로 설정
    returns.iloc[0] = 0
    
    # 리밸런싱 정보를 저장할 리스트 초기화
    rebalance_log = []
    # 초기 비중 기록
    rebalance_log.append({
        'date': returns.index[0],
        'weights': weights.copy()
    })
    
    # 리밸런싱 주기에 따라 날짜 분할
    if rebalance_frequency == 'none' or weight_calculator is None:
        # 리밸런싱 없이 초기 비중으로 전체 기간 백테스트
        weight_array = np.array([weights.get(col, 0) for col in returns.columns])
        portfolio_returns = returns.dot(weight_array.T)
    else:
        # 리밸런싱 주기에 따라 날짜 분할
        rebalance_dates = get_rebalance_dates(returns.index, rebalance_frequency, config)
        
        # 각 리밸런싱 기간별 포트폴리오 수익률 계산
        portfolio_returns_list = []
        current_weights = weights.copy()
        last_processed_date = None # 마지막 처리 날짜 추적
        
        for i in range(len(rebalance_dates) - 1):
            start_idx = rebalance_dates[i]
            end_idx = rebalance_dates[i + 1]
            
            # 중복 없이 기간 슬라이싱
            if last_processed_date is None: # 첫 기간
                period_returns_slice = returns.loc[start_idx:end_idx]
            else:
                # 마지막 처리 날짜 다음부터 현재 종료 날짜까지 선택
                period_returns_slice = returns.loc[(returns.index > last_processed_date) & (returns.index <= end_idx)]

            if period_returns_slice.empty: # 슬라이스가 비어있으면 건너뛰기
                continue
            
            # 마지막 처리 날짜 업데이트
            last_processed_date = period_returns_slice.index[-1]
            
            # 현재 비중으로 포트폴리오 수익률 계산
            weight_array = np.array([current_weights.get(col, 0) for col in period_returns_slice.columns])
            period_portfolio_returns = period_returns_slice.dot(weight_array.T)
            
            # 결과 저장
            portfolio_returns_list.append(period_portfolio_returns)
            
            # 다음 리밸런싱 시점에서 새로운 비중 계산 (마지막 기간 이후에는 불필요)
            if i < len(rebalance_dates) - 2:
                next_rebalance_date = rebalance_dates[i + 1]
                
                # 비중 계산 예외 처리
                try:
                    if lookback_period is not None:
                        # lookback_period 이전 데이터 사용
                        lookback_end_date = next_rebalance_date - pd.Timedelta(days=1)
                        lookback_returns = returns.loc[:lookback_end_date].iloc[-lookback_period:]
                        
                        # lookback 기간만큼 데이터가 있는지 확인
                        warning_result = handle_rebalancing_warning(
                            len(lookback_returns) >= lookback_period,
                            next_rebalance_date,
                            "lookback",
                            current_weights,
                            config
                        )
                        
                        if warning_result['has_warning']:
                            rebalance_log.append({
                                'date': next_rebalance_date,
                                'weights': warning_result['weights']
                            })
                        else:
                            current_weights = weight_calculator(lookback_returns, lookback_period=lookback_period)
                            rebalance_log.append({
                                'date': next_rebalance_date,
                                'weights': current_weights.copy()
                            })
                    else:
                        # lookback_period가 없으면 리밸런싱 날짜 이전 전체 데이터 사용
                        lookback_end_date = next_rebalance_date - pd.Timedelta(days=1)
                        lookback_returns = returns.loc[:lookback_end_date]
                        
                        warning_result = handle_rebalancing_warning(
                            not lookback_returns.empty,
                            next_rebalance_date,
                            "data",
                            current_weights,
                            config
                        )
                        
                        if warning_result['has_warning']:
                            rebalance_log.append({
                                'date': next_rebalance_date,
                                'weights': warning_result['weights']
                            })
                        else:
                            current_weights = weight_calculator(lookback_returns)
                            rebalance_log.append({
                                'date': next_rebalance_date,
                                'weights': current_weights.copy()
                            })
                except Exception as e:
                    # 에러 발생 시 이전 비중 유지
                    logging.error(f"Error calculating weights at {next_rebalance_date}: {e}. Keeping previous weights.")
                    rebalance_log.append({
                        'date': next_rebalance_date,
                        'weights': current_weights.copy()
                    })
        
        # 모든 기간의 포트폴리오 수익률 연결
        if portfolio_returns_list:
            portfolio_returns = pd.concat(portfolio_returns_list)
        else: # 리스트가 비어있는 경우 처리 (예: 데이터 기간이 매우 짧은 경우)
             portfolio_returns = pd.Series(dtype=float, index=pd.to_datetime([]))
    
    # 누적 수익률 계산 (cumprod 사용)
    cumulative_returns = (1 + portfolio_returns).cumprod() - 1
    
    # 성과 지표 계산
    performance_df = calculate_performance_metrics(portfolio_returns)
    
    return portfolio_returns, performance_df, rebalance_log

def initialize_portfolios(returns):
    """모든 포트폴리오 모델의 초기 비중을 계산합니다."""
    portfolio_weights = {}
    
    # 1. 균등 비중 포트폴리오
    equal_weight = CONFIG['portfolios']['equal_weight']['weight_per_asset']
    manual_weights = {symbol: equal_weight for symbol in CONFIG['symbols']}
    portfolio_weights[CONFIG['portfolios']['equal_weight']['name']] = manual_weights
    
    # 2. Risk Parity 포트폴리오
    risk_parity_weights_dict = risk_parity_weights(
        returns,
        lookback_period=CONFIG['optimization']['lookback_period'],
        min_weight=CONFIG['optimization']['min_weight'],
        max_weight=CONFIG['optimization']['max_weight'],
        config=CONFIG
    )
    portfolio_weights[CONFIG['portfolios']['risk_parity']['name']] = risk_parity_weights_dict
    
    # 3. 새로운 모델 (추가 시)
    # new_model_weights_dict = new_model_weights(...)
    # portfolio_weights[CONFIG['portfolios']['new_model']['name']] = new_model_weights_dict
    
    return portfolio_weights

# 메인 함수
def main():
    # 로깅 설정
    logger = setup_logging()
    
    # 데이터 로드
    stock_data = get_stock_data(CONFIG['symbols'])
    
    # 데이터 전처리
    returns = stock_data.pct_change()
    returns.iloc[0] = 0  # 첫날의 수익률을 0으로 설정
    
    # 모든 포트폴리오 초기 비중 계산
    portfolio_weights = initialize_portfolios(returns)
    
    # 각 포트폴리오에 대한 백테스트 실행 및 결과 저장
    results = {}
    portfolio_returns_dict = {}  # 포트폴리오별 수익률 저장
    rebalancing_info = {}  # 포트폴리오별 리밸런싱 정보 저장
    
    # 각 포트폴리오에 대해 연간 리밸런싱으로 백테스트 실행
    for portfolio_name, weights in portfolio_weights.items():
        portfolio_key = f"{portfolio_name} (연간 리밸런싱)"
        
        # 리밸런싱 시 비중 계산 함수 설정
        weight_calculator = None
        if portfolio_name == CONFIG['portfolios']['risk_parity']['name']:
            # Risk Parity 방식으로 비중 계산 (비중 제약 적용)
            weight_calculator = create_weight_calculator('risk_parity', config=CONFIG)
        else:
            # 수동 비중은 리밸런싱 시에도 동일하게 유지
            weight_calculator = create_weight_calculator('equal_weight', weights, CONFIG)
        
        # 백테스트 실행
        portfolio_returns, performance_df, rebalance_log = light_backtest(
            stock_data, 
            weights, 
            rebalance_frequency=CONFIG['rebalancing']['frequency'],
            weight_calculator=weight_calculator,
            lookback_period=CONFIG['optimization']['lookback_period'],
            config=CONFIG
        )
        
        # 결과 저장
        results[portfolio_key] = {
            'returns': portfolio_returns,
            'performance': performance_df.loc['Total Period']
        }
        
        # 포트폴리오 수익률 저장
        portfolio_returns_dict[portfolio_key] = portfolio_returns.values.flatten()
        
        # 리밸런싱 정보 저장
        rebalancing_info[portfolio_key] = rebalance_log
    
    # 포트폴리오별 수익률을 날짜를 인덱스로 하는 DataFrame으로 변환
    portfolio_returns_df = pd.DataFrame(portfolio_returns_dict)
    
    # 포트폴리오 비교
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
    
    # 최종 결과 시각화
    fig = plot_portfolio_return_comparison(portfolio_returns_df, comparison_df, rebalancing_info)
    fig.show() # 생성된 비교 그래프를 표시합니다.

if __name__ == "__main__":
    main()
    
