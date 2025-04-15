#!/usr/bin/env .venv/bin/python3.10
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import logging

class LinUCB:
    """LinUCB 알고리즘 구현 클래스"""
    def __init__(self, n_features, n_assets, alpha=0.1):
        self.n_features = n_features
        self.n_assets = n_assets
        self.alpha = alpha  # 탐색 파라미터
        
        # 각 자산별 선형 모델 파라미터 초기화
        self.A = {i: np.eye(n_features) for i in range(n_assets)}  # 특성 행렬
        self.b = {i: np.zeros(n_features) for i in range(n_assets)}  # 보상 벡터
        self.theta = {i: np.zeros(n_features) for i in range(n_assets)}  # 가중치 벡터
    
    def get_context(self, context_features, t):
        """컨텍스트 특성을 벡터로 변환"""
        return np.array([
            context_features['market_return'].iloc[t],
            context_features['market_volatility'].iloc[t],
            context_features['correlation'].iloc[t]
        ])
    
    def update(self, asset_idx, context, reward):
        """선형 모델 파라미터 업데이트"""
        # context를 2D 배열로 변환 (n_features, 1)
        context = context.reshape(-1, 1)
        
        # A 업데이트: context @ context.T는 (n_features, n_features) 행렬
        self.A[asset_idx] += context @ context.T
        
        # b 업데이트: reward는 스칼라, context는 (n_features, 1)
        self.b[asset_idx] += reward * context.flatten()
        
        # theta 업데이트: (n_features, n_features) @ (n_features,) = (n_features,)
        self.theta[asset_idx] = np.linalg.inv(self.A[asset_idx]) @ self.b[asset_idx]
    
    def predict(self, context):
        """각 자산의 예상 보상과 상한 신뢰 구간 계산"""
        # context를 2D 배열로 변환 (n_features, 1)
        context = context.reshape(-1, 1)
        predictions = {}
        
        for i in range(self.n_assets):
            # 예상 보상: theta는 (n_features,), context는 (n_features, 1)
            expected_reward = float(self.theta[i] @ context)
            
            # 상한 신뢰 구간: context.T는 (1, n_features), A_inv는 (n_features, n_features)
            confidence = self.alpha * np.sqrt(float(context.T @ np.linalg.inv(self.A[i]) @ context))
            
            # UCB 점수
            predictions[i] = expected_reward + confidence
        
        return predictions

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

def cvar_weights(returns, confidence_level=0.95, lookback_period=None, min_weight=None, max_weight=None, target_return=None, config=None):
    """
    CVaR(Conditional Value at Risk) 최소화 방식으로 포트폴리오 비중을 최적화합니다.
    
    Args:
        returns (DataFrame): 자산별 수익률 데이터프레임
        confidence_level (float): CVaR 계산에 사용되는 신뢰수준 (기본값: 0.95)
        lookback_period (int): 최적화에 사용할 과거 데이터 기간
        min_weight (float): 각 자산의 최소 비중
        max_weight (float): 각 자산의 최대 비중
        target_return (float): 목표 수익률 (None인 경우 제약 없음)
        config (dict): 설정 정보
        
    Returns:
        dict: 최적화된 자산별 비중
    """
    # 설정값 기본값 처리
    if config is not None:
        if min_weight is None:
            min_weight = config['optimization']['min_weight']
        if max_weight is None:
            max_weight = config['optimization']['max_weight']
        if lookback_period is None:
            lookback_period = config['optimization']['lookback_period']
        if 'confidence_level' in config['optimization']:
            confidence_level = config['optimization']['confidence_level']
    
    # 과거 기간 설정
    if lookback_period is not None:
        returns = returns.iloc[-lookback_period:]
    
    n_assets = len(returns.columns)
    n_samples = len(returns)
    
    # 수익률 행렬로 변환
    R = returns.values
    
    # 목표 수익률 설정
    mean_returns = np.mean(R, axis=0)
    
    # CVaR 최적화 목적 함수
    def portfolio_cvar(weights):
        # 포트폴리오 수익률 계산
        portfolio_returns = np.dot(R, weights)
        
        # VaR 계산 (신뢰수준에 해당하는 분위수)
        var = np.percentile(portfolio_returns, 100 * (1 - confidence_level))
        
        # CVaR 계산 (VaR 이하 수익률의 평균)
        cvar = -np.mean(portfolio_returns[portfolio_returns <= var])
        
        return cvar
    
    # 제약 조건: 비중 합계 = 1, 각 자산 비중은 제약 범위 내
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 비중 합계 = 1
    ]
    
    # 목표 수익률 제약 추가 (설정된 경우)
    if target_return is not None:
        constraints.append(
            {'type': 'eq', 'fun': lambda x: np.dot(mean_returns, x) - target_return}
        )
    
    bounds = tuple((min_weight, max_weight) for _ in range(n_assets))  # 각 자산의 비중 제약
    
    # 초기 비중 설정 (균등 분배)
    initial_weights = np.array([1/n_assets] * n_assets)
    
    # 최적화 수행 - 목적 함수: CVaR 최소화
    result = minimize(
        portfolio_cvar,
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    # 최적화 상태 및 메시지 출력
    if not result.success:
        logging.warning(f"CVaR 최적화 실패: {result.message}")
        # 실패 시 균등 비중 반환
        optimized_weights = {symbol: 1/n_assets for symbol in returns.columns}
        return optimized_weights
    
    # 비중 합이 1이 되도록 정규화
    normalized_weights = result.x / np.sum(result.x)
    
    # 최적화된 비중을 딕셔너리로 변환
    optimized_weights = {symbol: weight for symbol, weight in zip(returns.columns, normalized_weights)}
    
    return optimized_weights

def contextual_bandit_weights(returns, context_features=None, lookback_period=None, min_weight=None, max_weight=None, config=None):
    """
    LinUCB 알고리즘을 사용하여 포트폴리오 비중을 최적화합니다.
    선형 모델을 사용하여 각 자산의 보상 함수를 추정하고, 상한 신뢰 구간을 기반으로 비중을 결정합니다.
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
        if context_features is not None:
            context_features = context_features.iloc[-lookback_period:]
    
    n_assets = len(returns.columns)
    
    # 컨텍스트 특성이 없는 경우 기본 특성 생성
    if context_features is None:
        # 각 시점별로 과거 데이터만 사용하여 컨텍스트 생성
        market_returns = pd.Series(index=returns.index)
        market_volatility = pd.Series(index=returns.index)
        correlation = pd.Series(index=returns.index)
        
        # 과거 20일 윈도우 사용
        window = 20
        
        for t in range(window, len(returns)):
            # 과거 데이터만 사용
            past_returns = returns.iloc[t-window:t]
            
            # 시장 수익률 (과거 20일 평균)
            market_returns.iloc[t] = past_returns.mean(axis=1).mean()
            
            # 시장 변동성 (과거 20일 표준편차)
            market_volatility.iloc[t] = past_returns.std(axis=1).mean()
            
            # 상관관계 (과거 20일 상관관계)
            correlation.iloc[t] = past_returns.corr().mean().mean()
        
        context_features = pd.DataFrame({
            'market_return': market_returns,
            'market_volatility': market_volatility,
            'correlation': correlation
        })
    
    # LinUCB 모델 초기화
    n_features = len(context_features.columns)
    linucb = LinUCB(n_features, n_assets)
    
    # 과거 데이터로 학습
    window = 20  # 과거 20일 윈도우 사용
    
    for t in range(window, len(returns)):
        # t 시점의 컨텍스트는 t-1 시점까지의 정보만 사용
        context = linucb.get_context(context_features, t)
        
        # t 시점의 수익률 (이미 알려진 정보)
        daily_returns = returns.iloc[t]
        
        # 각 자산별 보상 계산 및 업데이트
        for i, asset in enumerate(returns.columns):
            reward = daily_returns[asset]
            linucb.update(i, context, reward)
    
    # 현재 시점에서의 최적 비중 계산
    current_context = linucb.get_context(context_features, -1)
    predictions = linucb.predict(current_context)
    
    # 예측값으로 비중 계산
    total_reward = sum(predictions.values())
    if total_reward == 0:
        return {asset: 1/n_assets for asset in returns.columns}
    
    weights = {asset: predictions[i]/total_reward for i, asset in enumerate(returns.columns)}
    
    # 비중 제약 적용
    weights = {asset: max(min(weight, max_weight), min_weight) for asset, weight in weights.items()}
    
    # 비중 합이 1이 되도록 정규화
    total_weight = sum(weights.values())
    weights = {asset: weight/total_weight for asset, weight in weights.items()}
    
    return weights

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
    elif portfolio_type == 'cvar':
        return lambda returns, lookback_period=None: cvar_weights(
            returns,
            lookback_period=lookback_period,
            min_weight=config['optimization']['min_weight'],
            max_weight=config['optimization']['max_weight'],
            config=config
        )
    elif portfolio_type == 'contextual_bandit':
        return lambda returns, lookback_period=None: contextual_bandit_weights(
            returns,
            lookback_period=lookback_period,
            min_weight=config['optimization']['min_weight'],
            max_weight=config['optimization']['max_weight'],
            config=config
        )
    return None



