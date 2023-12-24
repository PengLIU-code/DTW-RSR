import os
import json
import numpy as np
import pandas as pd
import copy

class SectorPreprocessor:
    def __init__(self, data_path, market_name): 
        self.data_path = data_path
        # self.date_format = '%Y-%m-%d %H:%M:%S' # 日期格式
        self.market_name = market_name

    def generate_sector_relation(self, industry_ticker_file, selected_tickers_fname):
        # 读取选择的股票代码
        selected_tickers = pd.read_csv(os.path.join(self.data_path, selected_tickers_fname))['Ticker'].to_numpy()
        print('#tickers selected:', len(selected_tickers))

        # 创建股票代码到索引的映射
        ticker_index = {ticker: index for index, ticker in enumerate(selected_tickers)}
        print('print(selected_tickers):', selected_tickers)
        print('ticker_index:', ticker_index)

        # 读取行业和股票代码的对应关系
        industry_tickers_df = pd.read_csv(os.path.join(self.data_path, industry_ticker_file))
        industry_tickers = industry_tickers_df.groupby('GICS Sub-Industry')['Ticker'].apply(list).to_dict()
        print('#industries: ', len(industry_tickers))
        print(industry_tickers.keys())

        # 筛选有效的行业（行业内股票数大于1）
        valid_industry_index = {industry: index for index, industry in enumerate(industry_tickers) if len(industry_tickers[industry]) > 1}
        one_hot_industry_embedding = np.identity(len(valid_industry_index) + 1, dtype=int)
        print("valid_industry_index:", valid_industry_index)
        print("one_hot_industry_embedding.shape:", one_hot_industry_embedding.shape)

        # 初始化股票关系嵌入矩阵
        ticker_relation_embedding = np.zeros([len(selected_tickers), len(selected_tickers), len(valid_industry_index) + 1], dtype=int)
        
        # 构建股票之间的行业关系
        for industry, index in valid_industry_index.items():
            cur_ind_tickers = industry_tickers[industry]
            if len(cur_ind_tickers) <= 1:
                continue

            for i in range(len(cur_ind_tickers)):
                left_tic = cur_ind_tickers[i]
                if left_tic not in ticker_index:
                    continue
                left_tic_ind = ticker_index[left_tic]
                ticker_relation_embedding[left_tic_ind][left_tic_ind] = copy.copy(one_hot_industry_embedding[index])
                ticker_relation_embedding[left_tic_ind][left_tic_ind][-1] = 1

                for j in range(i + 1, len(cur_ind_tickers)):
                    right_tic = cur_ind_tickers[j]
                    if right_tic not in ticker_index:
                        continue
                    right_tic_ind = ticker_index[right_tic]
                    ticker_relation_embedding[left_tic_ind][right_tic_ind] = copy.copy(one_hot_industry_embedding[index])
                    ticker_relation_embedding[right_tic_ind][left_tic_ind] = copy.copy(one_hot_industry_embedding[index])

        # 自相关矩阵
        for i in range(len(selected_tickers)):
            ticker_relation_embedding[i][i][-1] = 1

        print(ticker_relation_embedding.shape)
        np.save(self.market_name + '_industry_relation', ticker_relation_embedding)

# 设置数据路径和市场名称
data_path = '/Users/liupeng/Desktop/DSAA 5020 Final Project/TGCN_with_latest_data/2023.12.14(Data acquisition& Preprocessing)'
market_name = 'NASDAQ'

# 创建处理器实例
processor = SectorPreprocessor(data_path, market_name)

# 调用方法生成行业关系矩阵
processor.generate_sector_relation('NASDAQ_SP500_industry_ticker.csv', 'sp500_tickers_gics_sectors_industry.csv')

