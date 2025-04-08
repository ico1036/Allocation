#!/usr/bin/env .venv/bin/python3.10
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import logging

def risk_parity_weights(returns, target_risk_contribution=None, lookback_period=None, min_weight=None, max_weight=None, config=None):
    """
    Risk Parity 방식으로 포트폴리오 비중을 최적화합니다.
    """
    # 설정값 기본값 처리
    if config is not None:
        if min_weight is None:
            min_weight = config['optimization']['min_weight']
        if max_weight is None:
            max_weight = config['optimization']['max_weight']
        if lookback_period is None:
            lookback_period = config['optimization']['lookback_period']
    
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
    
    # 제약 조건: 비중 합계 = 1, 각 자산 비중은 제약 범위 내
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 비중 합계 = 1
    ]
    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))  # 각 자산의 비중 제약
    
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
    
    # 최적화 상태 및 메시지 출력
    if not result.success:
        logging.warning(f"최적화 실패: {result.message}")
        # 실패 시 균등 비중 반환
        optimized_weights = {symbol: 1/n_assets for symbol in returns.columns}
        return optimized_weights
    
    # 비중 합이 1이 되도록 정규화
    normalized_weights = result.x / np.sum(result.x)
    
    # 최적화된 비중을 딕셔너리로 변환
    optimized_weights = {symbol: weight for symbol, weight in zip(returns.columns, normalized_weights)}
    
    return optimized_weights

def create_weight_calculator(portfolio_type, initial_weights=None, config=None):
    """포트폴리오 타입에 맞는 비중 계산 함수를 반환합니다."""
    if portfolio_type == 'equal_weight':
        return lambda returns, lookback_period=None: initial_weights
    elif portfolio_type == 'risk_parity':
        return lambda returns, lookback_period=None: risk_parity_weights(
            returns,
            lookback_period=lookback_period,
            min_weight=config['optimization']['min_weight'],
            max_weight=config['optimization']['max_weight'],
            config=config
        )
    return None

# 여기에 추가 포트폴리오 모델 함수 추가 가능 