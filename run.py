from dataloader import get_stock_data


if __name__ == "__main__":
    symbols = ['PLTR','NVDA','SGOV','COIN']
    stock_data = get_stock_data(symbols)
    print(stock_data)
    