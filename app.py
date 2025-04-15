#!/usr/bin/env .venv/bin/python3.10
import streamlit as st
import pandas as pd
import numpy as np
from dataloader import get_stock_data
from util import (calculate_performance_metrics, setup_logging, get_rebalance_dates, 
                 handle_rebalancing_warning)
from model import risk_parity_weights, cvar_weights, create_weight_calculator, contextual_bandit_weights
from run import light_backtest
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt  # 이 줄 제거
import logging
import matplotlib.dates as mdates

# 페이지 설정
st.set_page_config(
    page_title="마이너스 바이러스 백신",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 초기 설정 파라미터
@st.cache_data
def get_default_config():
    return {
        # 최적화 관련 설정
        'optimization': {
            'min_weight': 0.05,  # 최소 비중 (5%)
            'max_weight': 0.60,  # 최대 비중 (60%)
            'lookback_period': 126,  # 약 6개월 (거래일 기준)
            'confidence_level': 0.95,  # CVaR 계산 신뢰수준 (95%)
        },
        
        # 리밸런싱 관련 설정
        'rebalancing': {
            'frequency': 'yearly',  # 리밸런싱 주기 ('yearly', 'quarterly', 'monthly')
            'quarterly_months': [1, 4, 7, 10],  # 분기별 리밸런싱에 사용되는 월
        },
        
        # 포트폴리오 설정
        'portfolios': {
            'equal_weight': {
                'name': '균등 분산 예방주사',
            },
            'risk_parity': {
                'name': '맞춤형 백신 솔루션',
            },
            'cvar': {
                'name': '극단상황 대비 백신',
            },
            'contextual_bandit': {
                'name': '시장 상황 학습 백신',
                'context_features': {
                    'market_return': True,
                    'market_volatility': True,
                    'correlation': True,
                    'vix': False,
                    'interest_rate': False,
                }
            }
        }
    }

# 로깅 설정
logger = setup_logging()

# 사이드바 설정
st.sidebar.title("🧪 백신 제조 설정")

# 티커 입력
st.sidebar.header("💊 투자 종목 선택")
ticker_input = st.sidebar.text_area("투자할 종목 코드를 입력하세요 (쉼표로 구분)", "AAPL, MSFT, BND")
tickers = [ticker.strip() for ticker in ticker_input.split(",") if ticker.strip()]

# 최적화 설정 섹션 제거
min_weight = 0.05  # 기본값 5%
max_weight = 0.60  # 기본값 60%
lookback_period = 126  # 기본값 126일

# 리밸런싱 설정
st.sidebar.header("⚖️ 포트폴리오 리밸런싱 주기")
rebalance_options = {
    'yearly': '연간 리밸런싱 (매년 첫 거래일)', 
    'quarterly': '분기별 리밸런싱 (1, 4, 7, 10월 첫 거래일)', 
    'monthly': '월간 리밸런싱 (매월 첫 거래일)',
    'none': '리밸런싱 없음 (초기 설정 유지)'
}
rebalance_frequency = st.sidebar.selectbox(
    "리밸런싱 주기", 
    list(rebalance_options.keys()),
    format_func=lambda x: rebalance_options[x]
)

# 메인 페이지
st.title("🦠 마이너스 바이러스 백신")
st.caption("투자 손실을 예방하는 리스크 분산 포트폴리오 도우미")

# 티커 확인
if len(tickers) < 2:
    st.warning("최소 2개 이상의 티커를 입력해주세요.")
    st.stop()

# 설정 업데이트
config = get_default_config()
config['symbols'] = tickers
config['optimization']['min_weight'] = min_weight
config['optimization']['max_weight'] = max_weight
config['optimization']['lookback_period'] = lookback_period
config['rebalancing']['frequency'] = rebalance_frequency
config['portfolios']['equal_weight']['weight_per_asset'] = 1/len(tickers)

# 진행 상태 표시
try:
    stock_data = get_stock_data(tickers)
    returns = stock_data.pct_change()
    returns.iloc[0] = 0  # 첫날의 수익률을 0으로 설정
    annual_returns = (1 + returns).prod() ** (252/len(returns)) - 1
    annual_vols = returns.std() * np.sqrt(252)
    sharpe_ratios = annual_returns / annual_vols
    correlation = returns.corr()

    # 1. 균등 비중 포트폴리오
    equal_weight = config['portfolios']['equal_weight']['weight_per_asset']
    manual_weights = {symbol: equal_weight for symbol in tickers}
    
    # 2. Risk Parity 포트폴리오
    risk_parity_weights_dict = risk_parity_weights(
        returns,
        lookback_period=config['optimization']['lookback_period'],
        min_weight=config['optimization']['min_weight'],
        max_weight=config['optimization']['max_weight'],
        config=config
    )
    
    # 3. CVaR 최적화 포트폴리오
    cvar_weights_dict = cvar_weights(
        returns,
        confidence_level=config['optimization']['confidence_level'],
        lookback_period=config['optimization']['lookback_period'],
        min_weight=config['optimization']['min_weight'],
        max_weight=config['optimization']['max_weight'],
        config=config
    )
    
    # 4. Contextual Bandit 포트폴리오
    contextual_bandit_weights_dict = contextual_bandit_weights(
        returns,
        lookback_period=config['optimization']['lookback_period'],
        min_weight=config['optimization']['min_weight'],
        max_weight=config['optimization']['max_weight'],
        config=config
    )
    
    portfolio_weights = {
        config['portfolios']['equal_weight']['name']: manual_weights,
        config['portfolios']['risk_parity']['name']: risk_parity_weights_dict,
        config['portfolios']['cvar']['name']: cvar_weights_dict,
        config['portfolios']['contextual_bandit']['name']: contextual_bandit_weights_dict
    }
    
    # 백테스트 실행
    results = {}
    portfolio_returns_dict = {}
    rebalancing_info = {}
    
    for portfolio_name, weights in portfolio_weights.items():
        portfolio_key = f"{portfolio_name} ({rebalance_options[rebalance_frequency]})"
        
        # 리밸런싱 시 비중 계산 함수 설정
        if portfolio_name == config['portfolios']['risk_parity']['name']:
            weight_calculator = create_weight_calculator('risk_parity', config=config)
        elif portfolio_name == config['portfolios']['cvar']['name']:
            weight_calculator = create_weight_calculator('cvar', config=config)
        elif portfolio_name == config['portfolios']['contextual_bandit']['name']:
            weight_calculator = create_weight_calculator('contextual_bandit', config=config)
        else:
            weight_calculator = create_weight_calculator('equal_weight', weights, config)
        
        # 백테스트 실행
        portfolio_returns, performance_df, rebalance_log = light_backtest(
            stock_data, 
            weights, 
            rebalance_frequency=config['rebalancing']['frequency'],
            weight_calculator=weight_calculator,
            lookback_period=config['optimization']['lookback_period'],
            config=config
        )
        
        results[portfolio_key] = {
            'returns': portfolio_returns,
            'performance': performance_df.loc['Total Period']
        }
        
        # 날짜 인덱스를 유지하기 위해 values.flatten() 대신 원본 DataFrame 사용
        portfolio_returns_dict[portfolio_key] = portfolio_returns
        rebalancing_info[portfolio_key] = rebalance_log
    
    # 데이터프레임 변환 - portfolio_returns_dict 값들이 모두 Series나 DataFrame이므로 concat 사용
    portfolio_returns_df = pd.concat(portfolio_returns_dict, axis=1)
    
    # 열 인덱스가 2단계(MultiIndex)인 경우 1단계로 변경
    if isinstance(portfolio_returns_df.columns, pd.MultiIndex):
        portfolio_returns_df.columns = [col[0] for col in portfolio_returns_df.columns]
    
    # 디버깅 메시지 추가
    print(f"백테스트 결과 데이터프레임 생성 완료: {portfolio_returns_df.shape}")
    print(f"백테스트 결과 인덱스 타입: {type(portfolio_returns_df.index)}")
    print(f"백테스트 결과 첫 5개 인덱스: {portfolio_returns_df.index[:5].tolist()}")
    
    # 포트폴리오 비교 데이터 준비
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
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.set_index('포트폴리오', inplace=True)
    
    st.success("분석 완료!")  # status.update 대신 st.success 사용
except Exception as e:
    st.error(f"오류 발생: {str(e)}")
    st.stop()

# 백테스트 결과를 먼저 표시
st.header("💰 투자 수익 백테스트 결과")

# 백테스트 결과 시각화 함수 (plotly 사용)
def plot_backtest_results(portfolio_returns_df, comparison_df, stock_data=None, rebalancing_info=None):
    """
    간결한 백테스트 결과 시각화 그래프를 생성합니다.
    
    Parameters:
    -----------
    portfolio_returns_df : pandas.DataFrame
        포트폴리오별 일별 수익률
    comparison_df : pandas.DataFrame
        포트폴리오 성과 비교 데이터
    stock_data : pandas.DataFrame, optional
        원본 주가 데이터 (날짜 인덱스 복원용)
    rebalancing_info : dict, optional
        포트폴리오별 리밸런싱 정보
    """
    # 누적 수익률 계산
    cumulative_returns = (1 + portfolio_returns_df).cumprod() - 1
    
    # 데이터 확인용 메시지 (디버깅)
    print(f"누적 수익률 데이터 포인트 수: {len(cumulative_returns)}")
    print(f"인덱스 타입: {type(cumulative_returns.index)}")
    print(f"cumulative_returns 타입: {type(cumulative_returns)}")
    print(f"cumulative_returns shape: {cumulative_returns.shape}")
    print(f"cumulative_returns 컬럼: {cumulative_returns.columns.tolist()}")
    print(f"인덱스 첫 5개 값: {cumulative_returns.index[:5].tolist()}")
    print(f"인덱스 마지막 5개 값: {cumulative_returns.index[-5:].tolist()}")
    
    # 날짜 인덱스 확인
    is_datetime_index = isinstance(cumulative_returns.index, pd.DatetimeIndex)
    if not is_datetime_index:
        print("인덱스가 DatetimeIndex가 아닙니다. 변환을 시도합니다.")
        try:
            # YYMMDD 형식의 문자열 인덱스를 datetime으로 변환
            cumulative_returns.index = pd.to_datetime(cumulative_returns.index, format='%y%m%d')
            is_datetime_index = True
            print(f"인덱스 변환 완료, 새 타입: {type(cumulative_returns.index)}")
            print(f"변환된 첫 5개 인덱스: {cumulative_returns.index[:5].tolist()}")
        except Exception as e:
            print(f"인덱스 변환 실패: {str(e)}")
    
    # 인덱스 값 중복/정렬 확인
    print(f"인덱스 값 중복 여부: {cumulative_returns.index.duplicated().any()}")
    print(f"인덱스 정렬 상태: {cumulative_returns.index.is_monotonic_increasing}")
    
    # 데이터 샘플 확인
    print("cumulative_returns 첫 3행:")
    print(cumulative_returns.head(3))
    
    # 포트폴리오 색상 설정
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # 그래프 생성
    fig = go.Figure()
    
    # 포트폴리오별 누적 수익률 선 추가
    for i, col in enumerate(cumulative_returns.columns):
        color_idx = i % len(colors)
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns[col],
                mode='lines',
                name=col,
                line=dict(color=colors[color_idx], width=2)
            )
        )
    
    # 레이아웃 설정
    fig.update_layout(
        title={
            'text': '예방접종 후 투자 면역력 비교',
            'font': {'size': 20, 'color': '#333'}
        },
        xaxis_title='날짜',
        yaxis_title='누적 수익률',
        yaxis_tickformat='.1%',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template='plotly_white',
        hovermode='x unified',
        margin=dict(l=60, r=30, t=80, b=60)
    )
    
    # X축 날짜 설정 (최소한의 설정만)
    fig.update_xaxes(
        type='date',
        tickangle=45,
        tickfont=dict(size=10)
    )
    
    return fig

# 결과 그래프 표시
fig = plot_backtest_results(portfolio_returns_df, comparison_df, rebalancing_info)
st.plotly_chart(fig, use_container_width=True)

# 종목별 통계 표시
st.header("🔬 종목별 면역력 분석")
with st.expander("📊 개별 종목 면역력 지표 (클릭하여 펼치기)", expanded=False):
    st.caption("각 종목의 개별 면역력 지표입니다.")
    col1, col2 = st.columns(2)

    with col1:
        stats_df = pd.DataFrame({
            '연간 수익률': annual_returns,
            '연간 변동성': annual_vols,
            '샤프 비율': sharpe_ratios
        })
        st.dataframe(
            stats_df.style.format({
                '연간 수익률': '{:.2%}', 
                '연간 변동성': '{:.2%}', 
                '샤프 비율': '{:.2f}'
            }), 
            use_container_width=True
        )
        st.caption("• 연간 수익률: 백신 투여 후 1년간 기대 수익률")
        st.caption("• 연간 변동성: 백신의 안정성 지표 (낮을수록 부작용 적음)")
        st.caption("• 샤프 비율: 백신 효능 점수 (높을수록 효과 좋음)")

    with col2:
        st.write("종목간 상호작용")
        st.dataframe(correlation.style.background_gradient(cmap='coolwarm'), use_container_width=True)
        st.caption("상호작용이 낮은 종목들을 조합하면 포트폴리오 면역력이 강화됩니다.")
        st.caption("• 1에 가까울수록: 같은 증상 보임 (함께 오르내림)")
        st.caption("• -1에 가까울수록: 반대 증상 보임 (하나가 오르면 다른 하나는 내림)")

# 성과 지표
st.subheader("📈 백신 효과 비교")
st.dataframe(
    comparison_df.style.format({
        '누적 수익률': '{:.2%}',
        '연간 수익률': '{:.2%}',
        '연간 변동성': '{:.2%}',
        '샤프 비율': '{:.2f}'  # 샤프비율은 퍼센트가 아닌 실수로 표시
    }),
    use_container_width=True
)
st.caption("• 누적 수익률: 백신 투여 후 전체 기간 총 수익")
st.caption("• 연간 수익률: 연간 기대 백신 효과")
st.caption("• 연간 변동성: 백신 안정성 지표 (낮을수록 부작용 적음)")
st.caption("• 샤프 비율: 백신 효능 점수 (높을수록 효과 좋음)")

# 리밸런싱 정보 (plotly가 사용 가능한 경우에만 표시)
if rebalance_frequency != 'none':
    st.header("⚖️ 리밸런싱 이력")
    
    # 리밸런싱 데이터 정리
    rebalance_data = []
    
    for portfolio_name, rebalance_log in rebalancing_info.items():
        for entry in rebalance_log:
            date = entry['date']
            weights = entry['weights']
            
            for symbol, weight in weights.items():
                rebalance_data.append({
                    '날짜': date,
                    '포트폴리오': portfolio_name,
                    '종목': symbol,
                    '비중': weight
                })
    
    rebalance_df = pd.DataFrame(rebalance_data)
    
    # 날짜별 리밸런싱 정보 표시 (tabs 사용)
    unique_dates = sorted(rebalance_df['날짜'].unique())
    st.subheader(f"리밸런싱 기록 ({len(unique_dates)}회)")
    
    # 날짜 탭 생성
    tabs = st.tabs([date.strftime('%Y-%m-%d') for date in unique_dates])
    
    # 각 탭에 데이터 채우기
    for i, date in enumerate(unique_dates):
        with tabs[i]:
            date_data = rebalance_df[rebalance_df['날짜'] == date]
            # pivot 대신 pivot_table 사용하여 중복 인덱스 문제 해결
            pivot_data = date_data.pivot_table(index='종목', columns='포트폴리오', values='비중', aggfunc='first')
            st.dataframe(pivot_data.style.format('{:.2%}'), use_container_width=True)
            
            # 추가 정보 표시
            st.caption(f"리밸런싱 일자: {date.strftime('%Y년 %m월 %d일')}")

# 포트폴리오 구성 탭을 가장 마지막에 배치
st.header("📋 마이 포트폴리오 우렁각시")
with st.expander("💼 최적 투자 비중 (클릭하여 펼치기)", expanded=True):
    # 최근 날짜 기준 비중 표시
    st.caption("실제 투자에 적용할 최적의 투자 비중입니다.")
    
    if rebalance_frequency != 'none' and 'rebalance_df' in locals():
        # 가장 최근 리밸런싱 날짜 찾기 (리밸런싱 이력과 동일)
        latest_date = max(rebalance_df['날짜'].unique())
        latest_date_str = latest_date.strftime('%Y년 %m월 %d일')
        
        # 최근 리밸런싱 데이터 필터링 (리밸런싱 이력 탭과 정확히 동일한 데이터)
        latest_rebalance_df = rebalance_df[rebalance_df['날짜'] == latest_date]
        
        # 정보 표시
        st.info(f"🔄 기준일: {latest_date_str} (가장 최근 리밸런싱 날짜)")
        
        # 포트폴리오별 테이블 표시 - 리밸런싱 탭과 동일한 형식
        st.subheader("🔮 우렁각시의 비중 조언")
        
        # 리밸런싱 이력과 완전히 동일한 pivot_table 사용
        pivot_data = latest_rebalance_df.pivot_table(index='종목', columns='포트폴리오', values='비중', aggfunc='first')
        st.dataframe(pivot_data.style.format('{:.2%}'), use_container_width=True)
        
    else:
        # 리밸런싱 없는 경우 초기 비중 사용
        latest_date_str = stock_data.index[-1].strftime('%Y년 %m월 %d일') 
        st.info(f"📅 기준일: {latest_date_str} (백테스트 마지막 날짜)")
        
        # 균등 비중과 Risk Parity 비중 초기값 테이블로 표시
        combined_weights = {}
        for symbol in tickers:
            combined_weights[symbol] = {
                config['portfolios']['equal_weight']['name']: manual_weights[symbol],
                config['portfolios']['risk_parity']['name']: risk_parity_weights_dict[symbol],
                config['portfolios']['cvar']['name']: cvar_weights_dict[symbol],
                config['portfolios']['contextual_bandit']['name']: contextual_bandit_weights_dict[symbol]
            }
        
        # 데이터프레임으로 변환
        combined_df = pd.DataFrame(combined_weights).T
        st.dataframe(combined_df.style.format('{:.2%}'), use_container_width=True)
        st.caption("리밸런싱 없이 초기 비중을 계속 유지합니다.")
            
    # 투자 가이드 추가
    st.success("💡 투자 가이드: 위 비중으로 투자하면 마이너스 바이러스를 예방할 수 있습니다!") 