import argparse
from datetime import datetime
import copy
import json
import numpy as np
import operator
import os
import pandas as pd

Classfied_Tickers_path = '/Users/liupeng/Desktop/DSAA 5020 Final Project/TGCN_with_latest_data/2023.12.14(Data acquisition& Preprocessing)/classified_by_gics_sub_industry_sp500_tickers.csv'
SP500tickers_path = "/Users/liupeng/Desktop/DSAA 5020 Final Project/TGCN_with_latest_data/2023.12.14(Data acquisition& Preprocessing)/sp500_tickers_gics_sectors_industry.csv"

# 该类用于生成行业关系矩阵，即每个股票代码对应的行业关系
class SectorPreprocessor:
    def __init__(self, data_path, market_name): 
        self.data_path = data_path
        self.date_format = '%Y-%m-%d %H:%M:%S' # 日期格式
        self.market_name = market_name
# 该函数用于生成行业关系矩阵，即每个股票代码对应的行业关系
    def generate_sector_relation(self, industry_ticker_file,
                                 selected_tickers_fname):
        selected_tickers = np.genfromtxt(
            os.path.join(self.data_path, selected_tickers_fname),
            dtype=str, delimiter='\t', skip_header=False
        )
        print('#tickers selected:', len(selected_tickers))
        ticker_index = {}
        print('print(selected_tickers):', selected_tickers)
        for index, ticker in enumerate(selected_tickers):
            ticker_index[ticker] = index
        print('ticker_index:',ticker_index)
        with open(industry_ticker_file, 'r') as fin:
            industry_tickers = json.load(fin)
        print('#industries: ', len(industry_tickers))
        print(industry_tickers.keys())
        valid_industry_count = 0
        valid_industry_index = {}
        for industry in industry_tickers.keys():
            # 行业内股票数大于1的行业才有意义，不然无法构建股票之间的关联
            if len(industry_tickers[industry]) > 1:
                valid_industry_index[industry] = valid_industry_count
                valid_industry_count += 1
        one_hot_industry_embedding = np.identity(valid_industry_count + 1,
                                                 dtype=int)
        print('-------------------------------------------------------------------------------------------------------')
        print("valid_industry_index:", valid_industry_index)
        # print("one_hot_industry_embedding:" , one_hot_industry_embedding)
        print("one_hot_industry_embedding.shape:", one_hot_industry_embedding.shape)
        ticker_relation_embedding = np.zeros(
            [len(selected_tickers), len(selected_tickers),
             valid_industry_count + 1], dtype=int) # 第三个维度一共有96个有效行业，加一是为了最后一个维度用于构建自相关矩阵
        print(ticker_relation_embedding[0][0].shape)
        print(industry_tickers['EDP Services'])
        # 如果行业中的股票在ticker中不存在，那么就把这个股票过滤掉
        for industry in valid_industry_index.keys():
            cur_ind_tickers = industry_tickers[industry]
            # 不懂为什么又重新筛选一遍无效行业，前面不是已经筛选过了吗？ n/a 行业目前还没过滤掉
            if len(cur_ind_tickers) <= 1:
                print('shit industry:', industry)
                continue
            # 这个嵌套循环遍历同一行业内的所有股票代码。对于每对股票代码，它会在矩阵中更新它们之间的关系。
            # left_tic_ind 和 right_tic_ind 是当前行业内两个不同股票代码的索引。
            # 关系矩阵被更新为这两个股票代码属于同一个行业的信息。这通过“one-hot”编码的方式实现，即在对应的位置上标记为1。
            ind_ind = valid_industry_index[industry]
            for i in range(len(cur_ind_tickers)):
                left_tic_ind = ticker_index[cur_ind_tickers[i]]
                ticker_relation_embedding[left_tic_ind][left_tic_ind] = \
                    copy.copy(one_hot_industry_embedding[ind_ind])
                ticker_relation_embedding[left_tic_ind][left_tic_ind][-1] = 1
                for j in range(i + 1, len(cur_ind_tickers)):
                    right_tic_ind = ticker_index[cur_ind_tickers[j]]
                    ticker_relation_embedding[left_tic_ind][right_tic_ind] = \
                        copy.copy(one_hot_industry_embedding[ind_ind])
                    ticker_relation_embedding[right_tic_ind][left_tic_ind] = \
                        copy.copy(one_hot_industry_embedding[ind_ind])
                    # print(right_tic_ind)

        # handle shit industry and n/a tickers
        for i in range(len(selected_tickers)):
            ticker_relation_embedding[i][i][-1] = 1
        print(ticker_relation_embedding.shape)
        print(ticker_relation_embedding[:,:,-2])
        non_zero_indices = np.nonzero(ticker_relation_embedding[:,:,-2])

        for index in zip(*non_zero_indices):
            print(f"位置: {index}, 值: {ticker_relation_embedding[index]}")

        np.save(self.market_name + '_industry_relation',
                ticker_relation_embedding)


if __name__ == '__main__':
    desc = "pre-process sector data market by market, including listing all " \
           "trading days, all satisfied stocks (5 years & high price), " \
           "normalizing and compansating data"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-path', help='path of EOD data')
    parser.add_argument('-market', help='market name')
    args = parser.parse_args()

    if args.path is None:
        args.path = '/Users/liupeng/Desktop/DSAA 5020 Final Project/TGCN_with_latest_data/Data'
    if args.market is None:
        args.market = 'NASDAQ'

    processor = SectorPreprocessor(args.path, args.market)

    processor.generate_sector_relation(
        os.path.join('/Users/liupeng/Desktop/DSAA 5020 Final Project/TGCN_with_latest_data/2023.12.14(Data acquisition& Preprocessing)/',
                     processor.market_name + '_SP500_industry_ticker.csv'),
        processor.market_name + '_tickers.csv'
    )