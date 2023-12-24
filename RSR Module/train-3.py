import random
import numpy as np
import os
import torch
from load_data import load_relation_data, load_EOD_data
from evaluator import evaluate
from model import get_loss, RelationLSTM, StockLSTM
import json


np.random.seed(123456789)
torch.random.manual_seed(12345678)
device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

data_path = '/content/drive/MyDrive/TGCN/TGCN_with_latest_data/TGCN_with_latest_data/Data'
market_name = 'NASDAQ'
relation_name = 'sector_industry'  # wikidata or sector_industry
parameters = {'seq': 5, 'unit': 64, 'alpha': 0.1}
epochs = 20
valid_index = 1112
test_index = 1588
fea_dim = 5
steps = 1

tickers_fname = market_name + '_tickers.csv'
tickers = np.genfromtxt(os.path.join(data_path, tickers_fname), dtype=str, delimiter='\t', skip_header=False)
batch_size = len(tickers)
print('#tickers selected:', len(tickers))
eod_data, mask_data, gt_data, price_data = load_EOD_data(data_path, market_name, tickers, steps)
trade_dates = mask_data.shape[1]
# relation data
rname_tail = {'sector_industry': '_industry_relation.npy', 'wikidata': '_wiki_relation.npy'}
rel_encoding, rel_mask = load_relation_data(
    os.path.join(data_path, market_name + rname_tail[relation_name])
)
print('relation encoding shape:', rel_encoding.shape)
print('relation mask shape:', rel_mask.shape)

model = RelationLSTM(
    batch_size=batch_size,
    rel_encoding=rel_encoding,
    rel_mask=rel_mask

).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
best_valid_loss = np.inf
best_test_loss = np.inf
best_valid_perf = None
best_test_perf = None
batch_offsets = np.arange(start=0, stop=valid_index, dtype=int)

# 初始化 records 字典
records = {
    'train_loss': [],
    'train_reg_loss': [],
    'train_rank_loss': [],
    'valid_loss': [],
    'valid_reg_loss': [],
    'valid_rank_loss': [],
    'valid_performance': [],
    'test_loss': [],
    'test_reg_loss': [],
    'test_rank_loss': [],
    'test_performance': []
}

def validate(start_index, end_index):
    """
    get loss on validate/test set
    """
    with torch.no_grad():
        cur_valid_pred = np.zeros([len(tickers), end_index - start_index], dtype=float)
        cur_valid_gt = np.zeros([len(tickers), end_index - start_index], dtype=float)
        cur_valid_mask = np.zeros([len(tickers), end_index - start_index], dtype=float)
        loss = 0.
        reg_loss = 0.
        rank_loss = 0.
        for cur_offset in range(start_index - parameters['seq'] - steps + 1, end_index - parameters['seq'] - steps + 1):
            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.Tensor(x).to(device),
                get_batch(cur_offset)
            )
            prediction = model(data_batch)
            cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                                     batch_size, parameters['alpha'])
            loss += cur_loss.item()
            reg_loss += cur_reg_loss.item()
            rank_loss += cur_rank_loss.item()
            cur_valid_pred[:, cur_offset - (start_index - parameters['seq'] - steps + 1)] = cur_rr[:, 0].cpu()
            cur_valid_gt[:, cur_offset - (start_index - parameters['seq'] - steps + 1)] = gt_batch[:, 0].cpu()
            cur_valid_mask[:, cur_offset - (start_index - parameters['seq'] - steps + 1)] = mask_batch[:, 0].cpu()
        loss = loss / (end_index - start_index)
        reg_loss = reg_loss / (end_index - start_index)
        rank_loss = rank_loss / (end_index - start_index)
        cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
    return loss, reg_loss, rank_loss, cur_valid_perf


def get_batch(offset=None):
    if offset is None:
        offset = random.randrange(0, valid_index)
    seq_len = parameters['seq']
    mask_batch = mask_data[:, offset: offset + seq_len + steps]
    mask_batch = np.min(mask_batch, axis=1)
    return (
        eod_data[:, offset:offset + seq_len, :],
        np.expand_dims(mask_batch, axis=1),
        np.expand_dims(price_data[:, offset + seq_len - 1], axis=1),
        np.expand_dims(gt_data[:, offset + seq_len + steps - 1], axis=1))


# train loop
best_daily_bt_data = None
for epoch in range(epochs):
    np.random.shuffle(batch_offsets)
    print("---------------------------------epoch:{}-----------------------------------".format(epoch))
    tra_loss = 0.0
    tra_reg_loss = 0.0
    tra_rank_loss = 0.0
    # steps
    for j in range(valid_index - parameters['seq'] - steps + 1): # valid_index - parameters['seq'] - steps + 1 实际上定义了在不超出训练数据范围的前提下，可以从数据中提取多少有效的时间序列样本。
        data_batch, mask_batch, price_batch, gt_batch = map(
            lambda x: torch.Tensor(x).to(device),
            get_batch(batch_offsets[j])
        )
        optimizer.zero_grad()
        prediction = model(data_batch)
        cur_loss, cur_reg_loss, cur_rank_loss, _ = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                            batch_size, parameters['alpha'])
        # update model
        cur_loss.backward()
        optimizer.step()

        tra_loss += cur_loss.item()
        tra_reg_loss += cur_reg_loss.item()
        tra_rank_loss += cur_rank_loss.item()
    # 在每个 epoch 结束时保存每日回测数据
    val_loss, val_reg_loss, val_rank_loss, epoch_perf = validate(valid_index, test_index)
    records['valid_performance'].append(epoch_perf)
    valid_daily_bt_data = {
            'daily_bt_long': epoch_perf['daily_bt_long'],
            'daily_bt_long5': epoch_perf['daily_bt_long5'],
            'daily_bt_long10': epoch_perf['daily_bt_long10']
        }
    with open('./records_error_detection/valid_model_daily_bt_data{}.json'.format(epoch), 'w') as f:
        f.write(json.dumps(valid_daily_bt_data))
    if val_loss < best_valid_loss:  # 使用 val_loss 替代 epoch_perf['mse']
        best_valid_loss = val_loss
        best_valid_perf = epoch_perf
        best_daily_bt_data = {
            'daily_bt_long': epoch_perf['daily_bt_long'],
            'daily_bt_long5': epoch_perf['daily_bt_long5'],
            'daily_bt_long10': epoch_perf['daily_bt_long10']
        }
        print('Better valid loss:', best_valid_loss)
    # train loss
    # loss = reg_loss(mse) + alpha*rank_loss
    tra_loss = tra_loss / (valid_index - parameters['seq'] - steps + 1)
    tra_reg_loss = tra_reg_loss / (valid_index - parameters['seq'] - steps + 1)
    tra_rank_loss = tra_rank_loss / (valid_index - parameters['seq'] - steps + 1)
    print('\n\nTrain : loss:{} reg_loss:{} rank_loss:{}'.format(tra_loss, tra_reg_loss, tra_rank_loss))

    # show performance on valid set
    # val_loss, val_reg_loss, val_rank_loss, val_perf = validate(valid_index, test_index)
    print('Valid : loss:{} reg_loss:{} rank_loss:{}'.format(val_loss, val_reg_loss, val_rank_loss))
    print('\t Valid performance:', epoch_perf)

    # show performance on valid set
    test_loss, test_reg_loss, test_rank_loss, test_perf = validate(test_index, trade_dates)
    print('Test: loss:{} reg_loss:{} rank_loss:{}'.format(test_loss, test_reg_loss, test_rank_loss))
    print('\t Test performance:', test_perf)
    test_daily_bt_data = {
            'Test_daily_bt_long': test_perf['daily_bt_long'],
            'Test_daily_bt_long5': test_perf['daily_bt_long5'],
            'Test_daily_bt_long10': test_perf['daily_bt_long10']
        }
    with open('./records_error_detection/test_daily_bt_data{}.json'.format(epoch), 'w') as f:
        f.write(json.dumps(test_daily_bt_data))

    # best result
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        # In this place, remove some var that wouldn't be printed
        # without copy.copy()
        # best_valid_perf = epoch_perf
        best_test_perf = test_perf
        best_test_daily_bt_data = {
            'Test_daily_bt_long': test_perf['daily_bt_long'],
            'Test_daily_bt_long5': test_perf['daily_bt_long5'],
            'Test_daily_bt_long10': test_perf['daily_bt_long10']
        }
        print('Better test loss:', best_test_loss)
    # 在每个 epoch 结束时更新 records 字典
    records['train_loss'].append(tra_loss)
    records['train_reg_loss'].append(tra_reg_loss)
    records['train_rank_loss'].append(tra_rank_loss)

    records['valid_loss'].append(val_loss)
    records['valid_reg_loss'].append(val_reg_loss)
    records['valid_rank_loss'].append(val_rank_loss)
    records['valid_performance'].append(epoch_perf)

    records['test_loss'].append(test_loss)
    records['test_reg_loss'].append(test_reg_loss)
    records['test_rank_loss'].append(test_rank_loss)
    records['test_performance'].append(test_perf)
    
print('\nBest Valid performance:', best_valid_perf)
print('Best Test performance:', best_test_perf)

with open('./records_error_detection/{}_records6.json'.format(type(model).__name__), 'w') as f:
    f.write(json.dumps(records))
with open('./records_error_detection/best_model_daily_bt_data6.json', 'w') as f:
    f.write(json.dumps(best_daily_bt_data))
with open('./records_error_detection/best_test_model_daily_bt_data6.json', 'w') as f:
    f.write(json.dumps(best_test_daily_bt_data))
