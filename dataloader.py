#!/usr/bin/env .venv/bin/python3.10
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_stock_data(symbols, n_years=10):
    """
    주어진 종목 코드들의 주가 데이터를 가져옵니다.
    
    Parameters:
    -----------
    symbols : list
        종목 코드 리스트 (예: ['005930.KS', '035720.KS'])
    n_years : int
        최소 필요 데이터 기간 (년)
    
    Returns:
    --------
    pandas.DataFrame
        날짜를 인덱스로, 종목을 컬럼으로 하는 DataFrame
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_years*252)
    
    result_data = {}
    excluded_symbols = []
    
    for symbol in symbols:
        try:
            # 개별 종목 데이터 다운로드
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)
            
            # 데이터 기간 체크
            if len(df) < 252 * 3:  # 252는 평균 거래일 수
                excluded_symbols.append(symbol)
                continue
            
            # Adj Close 데이터만 저장
            result_data[symbol] = df['Close']
            
        except Exception as e:
            excluded_symbols.append(symbol)
    
    # 모든 종목의 데이터를 하나의 DataFrame으로 결합
    if result_data:
        result_df = pd.DataFrame(result_data)
        # 날짜 인덱스를 YYMMDD 형식의 문자열로 변환
        result_df.index = result_df.index.strftime('%y%m%d')
        
        # NaN 값 처리
        result_df = result_df.dropna()
        
        return result_df
    else:
        return pd.DataFrame()

# 사용 예시
if __name__ == "__main__":
    # 테스트 케이스 1: 정상적인 종목들
    print("\n=== 테스트 케이스 1: 정상적인 종목들 ===")
    test_symbols_1 = [
        'AAPL',   # Apple (기술)
        'MSFT',   # Microsoft (기술)
        'JPM',    # JPMorgan Chase (금융)
        'JNJ',    # Johnson & Johnson (헬스케어)
        'XOM',    # Exxon Mobil (에너지)
        'PG',     # Procter & Gamble (소비재)
        'HD',     # Home Depot (소매)
        'UNH',    # UnitedHealth Group (헬스케어)
        'BAC',    # Bank of America (금융)
        'VZ'      # Verizon (통신)
    ]
    
    print("데이터 다운로드 시작...")
    stock_data_1 = get_stock_data(test_symbols_1)
    
    print("\n=== 데이터 미리보기 ===")
    print(stock_data_1.head())
    print("\n=== 데이터 정보 ===")
    print(stock_data_1.info())
    
    # 테스트 케이스 2: 잘못된 종목 코드 포함
    print("\n\n=== 테스트 케이스 2: 잘못된 종목 코드 포함 ===")
    test_symbols_2 = [
        'AAPL',   # Apple (정상)
        'MSFT',   # Microsoft (정상)
        'INVALID', # 잘못된 종목 코드
        'JNJ',    # Johnson & Johnson (정상)
        'WRONG',  # 잘못된 종목 코드
        'PG',     # Procter & Gamble (정상)
        'HD',     # Home Depot (정상)
        'UNH',    # UnitedHealth Group (정상)
        'BAC',    # Bank of America (정상)
        'VZ'      # Verizon (정상)
    ]
    
    print("데이터 다운로드 시작...")
    stock_data_2 = get_stock_data(test_symbols_2)
    
    print("\n=== 데이터 미리보기 ===")
    print(stock_data_2.head())
    print("\n=== 데이터 정보 ===")
    print(stock_data_2.info())
