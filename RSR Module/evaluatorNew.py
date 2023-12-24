import numpy as np
import csv
import datetime


def evaluate(prediction, ground_truth, mask, report=False):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
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
        rank_gt = np.argsort(ground_truth[:, i])
        gt_top1 = set()
        gt_top5 = set()
        gt_top10 = set()
        
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(gt_top1) < 1:
                gt_top1.add(cur_rank)
            if len(gt_top5) < 5:
                gt_top5.add(cur_rank)
            if len(gt_top10) < 10:
                gt_top10.add(cur_rank)
        rank_pre = np.argsort(prediction[:, i])
        pre_top1 = set()
        pre_top5 = set()
        pre_top10 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_pre[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(pre_top1) < 1:
                pre_top1.add(cur_rank)
            if len(pre_top5) < 5:
                pre_top5.add(cur_rank)
            if len(pre_top10) < 10:
                pre_top10.add(cur_rank)
        # 保存顶级 1 的索引及其收益率
        # top1_data.append((list(pre_top1), [ground_truth[idx][i] for idx in pre_top1]))

        # 保存顶级 5 的索引及其收益率
        # top5_rates = [ground_truth[idx][i] for idx in pre_top5]
        # top5_data.append((list(pre_top5), top5_rates))

        # 保存顶级 10 的索引及其收益率
        # top10_rates = [ground_truth[idx][i] for idx in pre_top10]
        # top10_data.append((list(pre_top10), top10_rates))
        
        # 保存顶级 1 的索引及其实际收益率和预测收益率
        top1_data.append((list(pre_top1), [ground_truth[idx][i] for idx in pre_top1], [prediction[idx][i] for idx in pre_top1]))

        # 保存顶级 5 的索引及其实际收益率和预测收益率
        top5_rates = [ground_truth[idx][i] for idx in pre_top5]
        top5_pred_rates = [prediction[idx][i] for idx in pre_top5]
        top5_data.append((list(pre_top5), top5_rates, top5_pred_rates))

        # 保存顶级 10 的索引及其实际收益率和预测收益率
        top10_rates = [ground_truth[idx][i] for idx in pre_top10]
        top10_pred_rates = [prediction[idx][i] for idx in pre_top10]
        top10_data.append((list(pre_top10), top10_rates, top10_pred_rates))

        # calculate mrr of top1
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

        # back testing on top 1
        real_ret_rat_top = ground_truth[list(pre_top1)[0]][i]
        bt_long += real_ret_rat_top

        # back testing on top 5
        real_ret_rat_top5 = 0
        for pre in pre_top5:
            real_ret_rat_top5 += ground_truth[pre][i]
        real_ret_rat_top5 /= 5
        bt_long5 += real_ret_rat_top5

        # back testing on top 10
        real_ret_rat_top10 = 0
        for pre in pre_top10:
            real_ret_rat_top10 += ground_truth[pre][i]
        real_ret_rat_top10 /= 10
        bt_long10 += real_ret_rat_top10

        # 单日回测数据收集
        daily_bt_long.append(real_ret_rat_top)
        daily_bt_long5.append(real_ret_rat_top5)
        daily_bt_long10.append(real_ret_rat_top10)
    filename = f"../records_error_detection(SP500)_DTW&New_Trading_Strategy(15Epoch,alpha=0.1)_1/top_stocks_with_returns_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    with open(filename, 'w', newline='') as file:
    # with open('./records_error_detection/top_stocks_with_returns.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Day', 'Top 1 Stocks and Returns', 'Top 5 Stocks and Returns', 'Top 10 Stocks and Returns'])
        for day in range(len(top1_data)):
            writer.writerow([day, top1_data[day], top5_data[day], top10_data[day]])

    filename = f"../records_error_detection(SP500)_DTW&New_Trading_Strategy(15Epoch,alpha=0.1)_1/top_stocks_with_returns_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        # 添加一个新的列头'Predicted Returns'
        writer.writerow(['Day', 'Top 1 Stocks and Returns', 'Top 5 Stocks and Returns', 'Top 10 Stocks and Returns', 'Predicted Returns'])
        for day in range(len(top1_data)):
            # 获取预测的收益率
            predicted_returns = prediction[day]
            # 写入预测的收益率
            writer.writerow([day, top1_data[day], top5_data[day], top10_data[day], predicted_returns])
    
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
