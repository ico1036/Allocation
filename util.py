#!/usr/bin/env .venv/bin/python3.10
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
        annual_return = float(((1 + year_returns).prod() - 1).iloc[0])
        
        # 연간 변동성 (일간 변동성 * sqrt(거래일 수))
        annual_volatility = float((year_returns.std() * np.sqrt(len(year_returns))).iloc[0])
        
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
    total_return = float(((1 + returns).prod() - 1).iloc[0])
    
    # CAGR (연평균 복합 성장률)
    cagr = (1 + total_return) ** (1 / total_years) - 1
    
    # 전체 기간 변동성
    total_volatility = float((returns.std() * np.sqrt(252)).iloc[0])  # 연율화 변동성
    
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
    fig.show() 



def plot_portfolio_return_comparison(portfolio_returns_df, comparison_df):
    """
    여러 포트폴리오의 누적 수익률을 비교하는 그래프와 성과 비교 테이블을 생성합니다.
    
    Parameters:
    -----------
    portfolio_returns_df : pandas.DataFrame
        포트폴리오별 일별 수익률 (날짜 인덱스, 포트폴리오 이름 컬럼)
    comparison_df : pandas.DataFrame
        포트폴리오 성과 비교 데이터
    """
    # 서브플롯 생성 (2행 1열)
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
    for column in cumulative_returns.columns:
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns[column],
                mode='lines',
                name=column,
                line=dict(width=2)
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
    
    # 그래프 레이아웃 설정
    fig.update_layout(
        height=800,  # 전체 높이 설정
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white'
    )
    
    # X축 날짜 형식 설정 (첫 번째 서브플롯)
    fig.update_xaxes(
        tickformat="%Y-%m-%d",
        tickangle=45,
        nticks=10,
        row=1, col=1
    )
    
    # Y축 설정 (첫 번째 서브플롯)
    fig.update_yaxes(
        title_text="누적 수익률 (%)",
        tickformat=".1%",
        row=1, col=1
    )
    
    # 그래프 표시
    fig.show()
    
    return fig