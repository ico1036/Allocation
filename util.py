#!/usr/bin/env .venv/bin/python3.10
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import sys

def calculate_performance_metrics(returns):
    """
    포트폴리오 수익률에 대한 성과 지표를 계산합니다.
    
    Parameters:
    -----------
    returns : pandas.Series
        포트폴리오 일별 수익률
    
    Returns:
    --------
    pandas.DataFrame
        성과 지표 DataFrame
    """
    # 연도별 데이터 추출
    returns.index = pd.to_datetime(returns.index, format='%y%m%d')
    returns_by_year = {}
    
    for year in returns.index.year.unique():
        year_returns = returns[returns.index.year == year]
        returns_by_year[year] = year_returns
    
    # 연도별 성과 지표 계산
    performance_data = {}
    
    for year, year_returns in returns_by_year.items():
        # 연간 수익률 (단일 값)
        annual_return = float(((1 + year_returns).prod() - 1))
        
        # 연간 변동성 (일간 변동성 * sqrt(거래일 수))
        annual_volatility = float((year_returns.std() * np.sqrt(len(year_returns))))
        
        # 샤프 비율 (무위험 수익률 0% 가정)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
        
        performance_data[year] = {
            'Annual Return': annual_return,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': sharpe_ratio
        }
    
    # 전체 기간 성과 지표 계산
    total_days = len(returns)
    total_years = total_days / 252  # 연간 평균 거래일 수 252일 가정
    
    # 전체 기간 수익률 (단일 값)
    total_return = float(((1 + returns).prod() - 1))
    
    # CAGR (연평균 복합 성장률)
    cagr = (1 + total_return) ** (1 / total_years) - 1
    
    # 전체 기간 변동성
    total_volatility = float((returns.std() * np.sqrt(252)))  # 연율화 변동성
    
    # 전체 기간 샤프 비율
    total_sharpe = (cagr) / total_volatility if total_volatility != 0 else 0
    
    # 전체 기간 성과 지표 추가
    performance_data['Total Period'] = {
        'Annual Return': cagr,  # CAGR을 연간 수익률로 사용
        'Annual Volatility': total_volatility,
        'Sharpe Ratio': total_sharpe
    }
    
    # DataFrame 생성
    performance_df = pd.DataFrame(performance_data).T
    
    return performance_df 

def visualize_backtest_results(cumulative_returns, returns, weights, stock_data, performance_df):
    """
    백테스트 결과를 시각화합니다.
    
    Parameters:
    -----------
    cumulative_returns : pandas.Series
        포트폴리오 누적 수익률
    returns : pandas.DataFrame
        일별 수익률 데이터
    weights : dict
        종목별 비중
    stock_data : pandas.DataFrame
        종목별 주가 데이터
    performance_df : pandas.DataFrame
        성과 지표 DataFrame
    """
  

    
    # Plotly를 사용한 시각화
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=('Portfolio Cumulative Returns', 'Performance Metrics'),
                        specs=[[{"type": "scatter"}], [{"type": "table"}]],
                        row_heights=[0.7, 0.3])
    
    # 포트폴리오 누적 수익률 추가
    fig.add_trace(
        go.Scatter(x=cumulative_returns.index, y=cumulative_returns.values.flatten(), 
                  name='Portfolio', line=dict(width=3, color='#1f77b4', shape='spline'),
                  hovertemplate='%{x}<br>Return: %{y:.2%}<extra></extra>'),
        row=1, col=1
    )
    
    # 개별 종목 누적 수익률 추가
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i, symbol in enumerate(weights.keys()):
        if symbol in stock_data.columns:
            color_idx = i % len(colors)
            stock_cumulative = (1 + returns[symbol]).cumprod() - 1
            fig.add_trace(
                go.Scatter(x=stock_cumulative.index, y=stock_cumulative.values, 
                          name=symbol, line=dict(width=1.5, color=colors[color_idx], shape='spline'),
                          hovertemplate='%{x}<br>Return: %{y:.2%}<extra></extra>'),
                row=1, col=1
            )
    
    # 성과 지표 테이블 추가
    fig.add_trace(
        go.Table(
            header=dict(
                values=['Period'] + list(performance_df.columns),
                fill_color='#1f77b4',
                font=dict(color='white', size=12),
                align='center'
            ),
            cells=dict(
                values=[performance_df.index] + [performance_df[col].apply(lambda x: f'{x:.2%}') for col in performance_df.columns],
                fill_color=[['#f7f7f7', 'white'] * (len(performance_df) // 2 + 1)],
                font=dict(size=11),
                align='center'
            )
        ),
        row=2, col=1
    )
    
    # 레이아웃 설정
    fig.update_layout(
        title_text='Portfolio Backtest Results',
        title_font=dict(size=20, color='#1f77b4'),
        height=800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        hovermode='x unified'
    )
    
    # X축 설정 (날짜 포맷 및 간격)
    fig.update_xaxes(
        title_text="Date",
        title_font=dict(size=12),
        tickfont=dict(size=10),
        gridcolor='lightgray',
        nticks=5,  # X축에 표시할 눈금 수 제한
        tickformat="%Y",  # YYYY 형식으로 표시
        row=1, col=1
    )
    
    # Y축 설정 (수익률 포맷)
    fig.update_yaxes(
        title_text="Cumulative Return",
        title_font=dict(size=12),
        tickfont=dict(size=10),
        gridcolor='lightgray',
        tickformat=".0%",
        row=1, col=1
    )
    
    # 그래프 표시
    # fig.show() # Removed to prevent duplicate display
    
    # return fig # Removed as it's not used



def plot_portfolio_return_comparison(portfolio_returns_df, comparison_df, rebalancing_info=None):
    """
    여러 포트폴리오의 누적 수익률을 비교하는 그래프와 성과 비교 테이블을 생성합니다.
    
    Parameters:
    -----------
    portfolio_returns_df : pandas.DataFrame
        포트폴리오별 일별 수익률 (날짜 인덱스, 포트폴리오 이름 컬럼)
    comparison_df : pandas.DataFrame
        포트폴리오 성과 비교 데이터
    rebalancing_info : dict, optional
        포트폴리오별 리밸런싱 정보 (날짜와 비중 정보)
    """
    # 리밸런싱 날짜 모음 (x축 눈금 표시용)
    all_rebalance_dates = set()
    
    # 리밸런싱 테이블 표시를 위한 데이터 준비
    if rebalancing_info:
        # 리밸런싱 데이터 준비
        rebalancing_dates = []
        portfolio_names = []
        weights_data = []
        
        for portfolio_name, rebalance_log in rebalancing_info.items():
            for rebalance_entry in rebalance_log:
                date = rebalance_entry['date']
                weights = rebalance_entry['weights']
                all_rebalance_dates.add(date)
                
                # 날짜 문자열로 변환
                date_str = date.strftime('%Y-%m-%d')
                rebalancing_dates.append(date_str)
                portfolio_names.append(portfolio_name)
                
                # 종목별 비중 정보 형식화
                weight_str = ", ".join([f"{symbol}: {weight:.4f}" for symbol, weight in weights.items()])
                weights_data.append(weight_str)
        
        # 3행 1열 서브플롯 구조 생성 (그래프, 성과 테이블, 리밸런싱 테이블)
        fig = make_subplots(
            rows=3, 
            cols=1,
            subplot_titles=('포트폴리오별 누적 수익률 비교', '포트폴리오 성과 비교', '리밸런싱 정보'),
            vertical_spacing=0.15,
            row_heights=[0.5, 0.2, 0.3],
            specs=[[{"type": "scatter"}], [{"type": "table"}], [{"type": "table"}]]
        )
    else:
        # 리밸런싱 정보가 없을 경우 2행 1열 구조
        fig = make_subplots(
            rows=2, 
            cols=1,
            subplot_titles=('포트폴리오별 누적 수익률 비교', '포트폴리오 성과 비교'),
            vertical_spacing=0.2,
            row_heights=[0.7, 0.3],
            specs=[[{"type": "scatter"}], [{"type": "table"}]]
        )
    
    # 누적 수익률 계산
    cumulative_returns = (1 + portfolio_returns_df).cumprod() - 1
    
    # 각 포트폴리오별 누적 수익률 추가
    colors = {
        '수동 균등 비중 (연간 리밸런싱)': 'blue',
        '제약있는 Risk Parity (연간 리밸런싱)': 'red'
    }
    
    # 모든 포트폴리오의 누적 수익률 추가
    for column in cumulative_returns.columns:
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns[column],
                mode='lines',
                name=column,
                line=dict(width=2, color=colors.get(column, 'blue'))
            ),
            row=1, col=1
        )
    
    # 인덱스를 컬럼으로 변환
    comparison_df_reset = comparison_df.reset_index()
    
    # 성과 비교 테이블 추가
    fig.add_trace(
        go.Table(
            header=dict(
                values=['포트폴리오', '누적 수익률', '연간 수익률', '연간 변동성', '샤프 비율'],
                fill_color='paleturquoise',
                align='center',
                font=dict(size=12, color='black')
            ),
            cells=dict(
                values=[
                    comparison_df_reset['포트폴리오'],
                    comparison_df_reset['누적 수익률'].apply(lambda x: f"{x:.2%}"),
                    comparison_df_reset['연간 수익률'].apply(lambda x: f"{x:.2%}"),
                    comparison_df_reset['연간 변동성'].apply(lambda x: f"{x:.2%}"),
                    comparison_df_reset['샤프 비율'].apply(lambda x: f"{x:.2f}")
                ],
                fill_color=[
                    'white',
                    ['#f0f0f0' if i % 2 == 0 else 'white' for i in range(len(comparison_df_reset))],
                    ['#f0f0f0' if i % 2 == 0 else 'white' for i in range(len(comparison_df_reset))],
                    ['#f0f0f0' if i % 2 == 0 else 'white' for i in range(len(comparison_df_reset))],
                    ['#f0f0f0' if i % 2 == 0 else 'white' for i in range(len(comparison_df_reset))]
                ],
                align='center',
                font=dict(size=11)
            )
        ),
        row=2, col=1
    )
    
    # 리밸런싱 테이블 추가 (리밸런싱 정보가 있는 경우에만)
    if rebalancing_info and len(rebalancing_dates) > 0:
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['리밸런싱 날짜', '포트폴리오', '종목별 비중'],
                    fill_color='lightgreen',
                    align='center',
                    font=dict(size=12, color='black')
                ),
                cells=dict(
                    values=[rebalancing_dates, portfolio_names, weights_data],
                    fill_color=['#f0f0f0', 'white', 'white'],
                    align='left',
                    font=dict(size=11)
                )
            ),
            row=3, col=1
        )
        
        # 그래프 높이 설정
        fig.update_layout(height=900)
    else:
        # 리밸런싱 정보가 없을 경우 높이 설정
        fig.update_layout(height=800)
    
    # 그래프 레이아웃 설정
    fig.update_layout(
        title_text="포트폴리오 연간 리밸런싱 성과 비교",
        title_font=dict(size=16),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white',
        hovermode='closest'
    )
    
    # 리밸런싱 날짜를 포함하여 x축 틱 설정
    if all_rebalance_dates:
        # 전체 날짜 범위 계산
        start_date = pd.to_datetime(portfolio_returns_df.index.min())
        end_date = pd.to_datetime(portfolio_returns_df.index.max())
        
        # 주요 날짜 포인트: 시작, 끝, 리밸런싱 날짜
        important_dates = [start_date, end_date] + list(pd.to_datetime(list(all_rebalance_dates)))
        important_dates.sort()
        
        # X축 날짜 형식 설정 (첫 번째 서브플롯)
        fig.update_xaxes(
            tickvals=important_dates,
            ticktext=[date.strftime('%Y-%m-%d') for date in important_dates],
            tickangle=45,
            tickfont=dict(size=10),
            row=1, col=1
        )
    else:
        # 리밸런싱 날짜가 없을 경우 기본 설정
        fig.update_xaxes(
            tickformat="%Y-%m-%d",
            tickangle=45,
            nticks=10,
            tickfont=dict(size=10),
            row=1, col=1
        )
    
    # Y축 설정 (첫 번째 서브플롯)
    fig.update_yaxes(
        title_text="누적 수익률 (%)",
        tickformat=".1%",
        row=1, col=1
    )
    
    return fig

def setup_logging(log_level=logging.INFO):
    """필수적인 로깅을 위한 설정"""
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # 이미 핸들러가 있는 경우 제거 (중복 출력 방지)
    if logger.handlers:
        logger.handlers.clear()
    
    # 콘솔 출력 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def handle_rebalancing_warning(condition, date, warning_type, current_weights, config=None):
    """리밸런싱 관련 경고를 처리하고 로깅합니다."""
    if not condition:
        message = ""
        if warning_type == "lookback":
            lookback_period = config['optimization']['lookback_period'] if config else None
            message = f"Insufficient data for lookback period {lookback_period} at {date}"
        elif warning_type == "data":
            message = f"Insufficient data before {date}"
        
        logging.warning(f"{message}. Keeping previous weights.")
        
        # 동일한 반환 형식 유지
        return {
            'has_warning': True,
            'weights': current_weights.copy()
        }
    
    return {'has_warning': False}

def get_rebalance_dates(dates, frequency=None, config=None):
    """
    리밸런싱 주기에 따라 날짜를 분할합니다.
    """
    # 설정값 기본값 처리
    if config is not None and frequency is None:
        frequency = config['rebalancing']['frequency']
    
    if frequency == 'yearly':
        # 연간 리밸런싱 (매년 첫 거래일)
        rebalance_dates = [dates[0]]  # 시작일 추가
        for year in range(dates[0].year, dates[-1].year + 1):
            year_dates = dates[dates.year == year]
            if not year_dates.empty:
                rebalance_dates.append(year_dates[0])
    elif frequency == 'monthly':
        # 월간 리밸런싱 (매월 첫 거래일)
        rebalance_dates = [dates[0]]  # 시작일 추가
        for year in range(dates[0].year, dates[-1].year + 1):
            for month in range(1, 13):
                month_dates = dates[(dates.year == year) & (dates.month == month)]
                if not month_dates.empty:
                    rebalance_dates.append(month_dates[0])
    elif frequency == 'quarterly':
        # 분기별 리밸런싱 (매 분기 첫 거래일)
        quarterly_months = config['rebalancing']['quarterly_months'] if config else [1, 4, 7, 10]
        rebalance_dates = [dates[0]]  # 시작일 추가
        for year in range(dates[0].year, dates[-1].year + 1):
            for quarter in quarterly_months:
                quarter_dates = dates[(dates.year == year) & (dates.month == quarter)]
                if not quarter_dates.empty:
                    rebalance_dates.append(quarter_dates[0])
    else:
        # 기본값: 리밸런싱 없음 (시작일과 종료일만 포함)
        rebalance_dates = [dates[0], dates[-1]]
    
    # 마지막 날짜가 포함되어 있지 않으면 추가
    if rebalance_dates[-1] != dates[-1]:
        rebalance_dates.append(dates[-1])
    
    return rebalance_dates