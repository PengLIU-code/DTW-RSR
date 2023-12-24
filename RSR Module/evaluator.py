import numpy as np
import csv
import datetime


def evaluate(prediction, ground_truth, mask, report=False):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    print('ground_truth.shape:',ground_truth.shape)
    print('prediction.shape:',prediction.shape)
    performance = {}
    # mse
    performance['mse'] = np.linalg.norm((prediction - ground_truth) * mask) ** 2 / np.sum(mask)
    mrr_top = 0.0
    all_miss_days_top = 0
    bt_long = 1.0
    bt_long5 = 1.0
    bt_long10 = 1.0

    # 初始化每日回测数据列表
    daily_bt_long = []
    daily_bt_long5 = []
    daily_bt_long10 = []

    top1_data = []
    top5_data = []
    top10_data = []


    for i in range(prediction.shape[1]):
      # '''
      # 使用 np.argsort 对第 i 列（天）的实际（ground truth）数据进行排序。
      # 这将返回一个索引数组，其中的索引按照对应值的升序排列。
      # 换句话说，它返回的是股票按实际收益率排序后的顺序。
      # '''
      rank_gt = np.argsort(ground_truth[:, i])
      # 初始化几个集合来存储前1、前5和前10名的股票索引：
      gt_top1 = set()
      gt_top5 = set()
      gt_top10 = set()
      # 这些集合将用于存储基于实际数据的前1、前5和前10名股票的索引
      pre_top1 = set()
      pre_top5 = set()
      pre_top10 = set()
      #  下面循环从 1 开始而不是通常的 0，主要是因为它与数组索引的负数配合使用，旨在从数组的末尾开始访问元素。
      for j in range(1, prediction.shape[0] + 1):
         # 这个循环遍历每只股票。prediction.shape[0] 是股票的数量
        cur_rank = rank_gt[-1 * j] # 从排序后的实际数据（rank_gt）中选取第 j 高的股票
          if mask[cur_rank][i] < 0.5: # 如果 mask 数组中对应的值小于0.5，则跳过当前股票
              continue
          if len(gt_top1) < 1:
              gt_top1.add(cur_rank)
          if len(gt_top5) < 5:
              gt_top5.add(cur_rank)
          if len(gt_top10) < 10:
              gt_top10.add(cur_rank)
      rank_pre = np.argsort(prediction[:, i]) # 对第 i 列的预测数据进行排序，以便找出预测的前1、前5和前10名股票
        
      for j in range(1, prediction.shape[0] + 1):
        cur_rank = rank_pre[-1 * j]
          if mask[cur_rank][i] < 0.5:
            continue
          if len(pre_top1) < 1 and ground_truth[cur_rank][i] > 0:
              pre_top1.add(cur_rank)
          if len(pre_top5) < 5:
              pre_top5.add(cur_rank)
          if len(pre_top10) < 10:
              pre_top10.add(cur_rank)
        # 保存顶级 1 的索引及其收益率
        top1_data.append((list(pre_top1), [ground_truth[idx][i] for idx in pre_top1]))

        # 保存顶级 5 的索引及其收益率
        top5_rates = [ground_truth[idx][i] for idx in pre_top5]
        top5_data.append((list(pre_top5), top5_rates))

        # 保存顶级 10 的索引及其收益率
        top10_rates = [ground_truth[idx][i] for idx in pre_top10]
        top10_data.append((list(pre_top10), top10_rates))
        
        # calculate mrr of top1 计算所谓的平均倒数排名（Mean Reciprocal Rank, MRR）
        top1_pos_in_gt = 0
        # for each stock rank 1 to 1026
        # got the real rank of prediction top 1
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            else:
                top1_pos_in_gt += 1
                if cur_rank in pre_top1:
                    break
        if top1_pos_in_gt == 0:
            all_miss_days_top += 1
        else:
            mrr_top += 1.0 / top1_pos_in_gt

        # 计算真实收益率，如果预测收益率为负，则不进行交易
        real_ret_rat_top = sum([ground_truth[pre][i] for pre in pre_top1]) / max(len(pre_top1), 1) if pre_top1 else 0
        real_ret_rat_top5 = sum([ground_truth[pre][i] for pre in pre_top5 if ground_truth[pre][i] > 0]) / 5
        real_ret_rat_top10 = sum([ground_truth[pre][i] for pre in pre_top10 if ground_truth[pre][i] > 0]) / 10
        
        # # back testing on top 1
        # real_ret_rat_top = ground_truth[list(pre_top1)[0]][i]
        bt_long += real_ret_rat_top

        # # back testing on top 5
        # real_ret_rat_top5 = 0
        # for pre in pre_top5:
        #     real_ret_rat_top5 += ground_truth[pre][i]
        # real_ret_rat_top5 /= 5
        bt_long5 += real_ret_rat_top5

        # # back testing on top 10
        # real_ret_rat_top10 = 0
        # for pre in pre_top10:
        #     real_ret_rat_top10 += ground_truth[pre][i]
        # real_ret_rat_top10 /= 10
        bt_long10 += real_ret_rat_top10

        # 单日回测数据收集
        daily_bt_long.append(real_ret_rat_top)
        daily_bt_long5.append(real_ret_rat_top5)
        daily_bt_long10.append(real_ret_rat_top10)

    print('len(top1_data):',len(top1_data))
    filename = f"../records_error_detection(SP500)_DTW&New_Trading_Strategy(20Epoch,alpha=1)/top_stocks_with_returns_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    with open(filename, 'w', newline='') as file:
    # with open('./records_error_detection/top_stocks_with_returns.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Day', 'Top 1 Stocks and Returns', 'Top 5 Stocks and Returns', 'Top 10 Stocks and Returns'])
        for day in range(len(top1_data)):
            writer.writerow([day, top1_data[day], top5_data[day], top10_data[day]])

    # 1/real position average
    performance['mrrt'] = mrr_top / (prediction.shape[1] - all_miss_days_top)
    # prediction best return ratio
    performance['btl'] = bt_long
    # prediction top 5 average ratio
    performance['btl5'] = bt_long5
    # top 10 average
    performance['btl10'] = bt_long10
    # 将每日回测数据添加到 performance 字典中
    performance['daily_bt_long'] = daily_bt_long
    performance['daily_bt_long5'] = daily_bt_long5
    performance['daily_bt_long10'] = daily_bt_long10
    return performance
