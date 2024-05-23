import torch
from torch.utils.data import DataLoader
import numpy as np
import time, os
import pickle
import copy
from setting import Setting
from trainer import FlashbackTrainer
from dataloader import PoiDataloader
from dataset import Split
from utils import *
from network import create_h0_strategy
from evaluation import Evaluation
from tqdm import tqdm
from scipy.sparse import coo_matrix
from collections import defaultdict, Counter
from model import EnsembleModel

from torch.optim.lr_scheduler import _LRScheduler

class PolynomialDecayLR(_LRScheduler):

    def __init__(self, optimizer, warmup_updates, tot_updates, lr, end_lr, power, last_epoch=-1, verbose=False):
        self.warmup_updates = warmup_updates
        self.tot_updates = tot_updates
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super(PolynomialDecayLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            self.warmup_factor = self._step_count / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
        elif self._step_count >= self.tot_updates:
            lr = self.end_lr
        else:
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_lr
            pct_remaining = 1 - (self._step_count - warmup) / (
                self.tot_updates - warmup
            )
            lr = lr_range * pct_remaining ** (self.power) + self.end_lr

        return [lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        assert False


def save_user_item_context_corrected(train_data, max_len):
    item_context = defaultdict(list)
    n_item_context = defaultdict(int)

    for sequences in train_data.sequences:
        for user_sequences in sequences:
            user_sequences = [x + 1 for x in user_sequences]
            for i, item in enumerate(user_sequences):
                # 对于每个地点，生成其上下文
                if i < max_len:
                    # 如果当前索引小于max_len，则需要填充0
                    context_seq = [0]*(max_len - i - 1) + user_sequences[:i+1]
                else:
                    # 否则取前max_len个地点作为上下文
                    context_seq = user_sequences[i + 1 - max_len:i + 1]
                
                item_context[item].append(context_seq)
                n_item_context[item] += 1
    return  item_context, n_item_context

def sort_item_context_by_count(item_context, n_item_context):
    # 根据每个key（地点）容纳的轨迹数量排序item_context
    sorted_item_context = dict(sorted(item_context.items(), key=lambda x: n_item_context[x[0]], reverse=True))

    # 保存每个key对应的数量，降序排列
    sorted_n_item_context = {k: n_item_context[k] for k in sorted_item_context.keys()}

    return sorted_item_context, sorted_n_item_context

def preprocess_expert_data(expert_data):
    """
    预处理一个专家的数据。
    :param expert_data: 一个专家的数据，为POI和对应序列的字典。
    :return: 处理后的Tensor格式数据。
    """
    processed_data = {}
    for poi, sequences in expert_data.items():
        # 将序列转换为Tensor
        processed_data[poi] = torch.tensor(sequences, dtype=torch.long)
    return processed_data

def process_all_experts(experts_data_list):
    """
    处理所有专家的数据。
    :param experts_data_list: 包含所有专家数据的列表。
    :return: 所有专家处理后的数据列表。
    """
    all_experts_processed = []
    for expert_data in experts_data_list:
        processed_data = preprocess_expert_data(expert_data)
        all_experts_processed.append(processed_data)
    return all_experts_processed

def compute_adjustment(label_freq, setting):
    """compute the base probabilities"""

    label_freq_array = np.array(list(label_freq.values()))
    max_freq = label_freq_array.max()
    tau = 1.2
    adjustments = tau * (1-(np.log(label_freq_array+1e-4) / np.log(max_freq+1e-4)))
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(setting.device)
    return adjustments

def check_for_nan(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN detected in gradients of parameter: {name}")


seed = 3407  # 或任何你选择的数字

random.seed(seed)  # 为Python内建的随机数生成器设置种子
np.random.seed(seed)  # 为NumPy的随机数生成器设置种子
torch.manual_seed(seed)  # 为Torch的随机数生成器设置种子


# parse settings
setting = Setting()
setting.parse()
dir_name = os.path.dirname(setting.log_file)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
setting.log_file = setting.log_file + '_' + timestring
setting.model_files = setting.model_file + '_' + timestring +'.pth'
log = open(setting.log_file, 'w')



# print(setting)

# log_string(log, 'log_file: ' + setting.log_file)
# log_string(log, 'user_file: ' + setting.trans_user_file)
# log_string(log, 'loc_temporal_file: ' + setting.trans_loc_file)
# log_string(log, 'loc_spatial_file: ' + setting.trans_loc_spatial_file)
# log_string(log, 'interact_file: ' + setting.trans_interact_file)

# log_string(log, str(setting.lambda_user))
# log_string(log, str(setting.lambda_loc))

# log_string(log, 'W in AXW: ' + str(setting.use_weight))
# log_string(log, 'GCN in user: ' + str(setting.use_graph_user))
# log_string(log, 'spatial graph: ' + str(setting.use_spatial_graph))

message = ''.join([f'{k}: {v}\n' for k, v in vars(setting).items()])
log_string(log, message)

# load dataset
# poi_loader = PoiDataloader(
#     setting.max_users, setting.min_checkins)  # 0， 5*20+1
# poi_loader.read(setting.dataset_file)

with open(setting.loader_file, 'rb') as f:
    poi_loader = pickle.load(f)
# print('Active POI number: ', poi_loader.locations())  # 18737 106994
# print('Active User number: ', poi_loader.user_count())  # 32510 7768
# print('Total Checkins number: ', poi_loader.checkins_count())  # 1278274
    


log_string(log, 'Active POI number:{}'.format(poi_loader.locations()))
log_string(log, 'Active User number:{}'.format(poi_loader.user_count()))
log_string(log, 'Total Checkins number:{}'.format(poi_loader.checkins_count()))

dataset = poi_loader.create_dataset(
    setting.sequence_length, setting.batch_size, Split.TRAIN)  # 20, 200 or 1024, 0
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
dataset_test = poi_loader.create_dataset(
    setting.sequence_length, setting.batch_size, Split.TEST)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
assert setting.batch_size < poi_loader.user_count(
), 'batch size must be lower than the amount of available users'



# create flashback trainer
with open(setting.trans_loc_file, 'rb') as f:  # transition POI graph
    transition_graph = pickle.load(f)  # 在cpu上
# transition_graph = top_transition_graph(transition_graph)
transition_graph = coo_matrix(transition_graph)

if setting.use_spatial_graph:
    with open(setting.trans_loc_spatial_file, 'rb') as f:  # spatial POI graph
        spatial_graph = pickle.load(f)  # 在cpu上
    # spatial_graph = top_transition_graph(spatial_graph)
    spatial_graph = coo_matrix(spatial_graph)
else:
    spatial_graph = None

if setting.use_graph_user:
    with open(setting.trans_user_file, 'rb') as f:
        friend_graph = pickle.load(f)  # 在cpu上
    # friend_graph = top_transition_graph(friend_graph)
    friend_graph = coo_matrix(friend_graph)
else:
    friend_graph = None

with open(setting.trans_interact_file, 'rb') as f:  # User-POI interaction graph
    interact_graph = pickle.load(f)  # 在cpu上
interact_graph = csr_matrix(interact_graph)

log_string(log, 'Successfully load graph')

trainer = FlashbackTrainer(setting.lambda_t, setting.lambda_s, setting.lambda_loc, setting.lambda_user,
                           setting.use_weight, transition_graph, spatial_graph, friend_graph, setting.use_graph_user,
                           setting.use_spatial_graph, interact_graph)  # 0.01, 100 or 1000


h0_strategy = create_h0_strategy(
    setting.hidden_dim, setting.is_lstm)  # 10 True or False
trainer.prepare(poi_loader.locations(), poi_loader.user_count(), setting.hidden_dim, setting.rnn_factory,
                setting.device, setting)

# trainer.model.load_state_dict(torch.load(setting.model_file+'_20240130214954.pth')['state_dict'])

evaluation_test = Evaluation(dataset_test, dataloader_test,
                             poi_loader.user_count(), h0_strategy, trainer, setting, log)
print('{} {}'.format(trainer, setting.rnn_factory))

logits = compute_adjustment(dataset.freq, setting)

# acc1 = evaluation_test.evaluate(logits, dataset)



#  training loop
optimizer = torch.optim.AdamW(trainer.parameters(
), lr=setting.learning_rate, weight_decay=setting.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[20, 40, 60, 80], gamma=0.2)
# scheduler = PolynomialDecayLR(optimizer, warmup_updates=60000 , tot_updates=100000, lr=setting.learning_rate, end_lr=1e-9, power=1)

param_count = trainer.count_parameters()
log_string(log, f'In total: {param_count} trainable parameters')

bar = tqdm(total=setting.epochs)
bar.set_description('Training')



best_acc = 0

for e in range(setting.epochs):  # 100
    h = h0_strategy.on_init(setting.batch_size, setting.device)
    dataset.shuffle_users()  # shuffle users before each epoch!

    losses = []
    epoch_start = time.time()

    for i, (x, t, t_slot, s, y, y_t, y_t_slot, y_s, reset_h, active_users, f, y_f) in enumerate(dataloader):
        # reset hidden states for newly added users


        x = x.squeeze().to(setting.device)
        # t = t.squeeze().to(setting.device)
        t_slot = t_slot.squeeze().to(setting.device)
        s = s.squeeze().to(setting.device)

        y = y.squeeze().to(setting.device)
        # y_t = y_t.squeeze().to(setting.device)
        y_t_slot = y_t_slot.squeeze().to(setting.device)
        # y_s = y_s.squeeze().to(setting.device)
        active_users = active_users.to(setting.device)

        f = f.squeeze().to(setting.device)
        y_f = y_f.squeeze().to(setting.device)

        # mgda = MGDA()

        optimizer.zero_grad()
        loss = trainer.loss(x, t, t_slot, s, y, y_t,
                            y_t_slot, y_s, h, active_users, f, y_f, logits, dataset)
        

        loss.backward(retain_graph=True)
        check_for_nan(trainer.model)
            
        # torch.nn.utils.clip_grad_norm_(trainer.parameters(), 5)
        losses.append(loss.item())
        optimizer.step()

    # schedule learning rate:
    scheduler.step()
    bar.update(1)
    epoch_end = time.time()
    log_string(log, 'One training need {:.2f}s'.format(
        epoch_end - epoch_start))
    # statistics:
    if (e + 1) % 1 == 0:
        epoch_loss = np.mean(losses)
        log_string(log, f'Epoch: {e + 1}/{setting.epochs}')
        log_string(log, f'Used learning rate: {scheduler.get_last_lr()[0]}')
        log_string(log, f'Avg Loss: {epoch_loss}')

    if (e + 1) % setting.validate_epoch == 0:
        log_string(log, f'~~~ Test Set Evaluation (Epoch: {e + 1}) ~~~')
        evl_start = time.time()
        acc1 = evaluation_test.evaluate(logits, dataset)
        if acc1 > best_acc:
            state = {
            'epoch': e,
            'state_dict': trainer.model.state_dict(),
            'optimizer': optimizer.state_dict(),
            # 如果你有使用学习率调度器
            'scheduler': scheduler.state_dict() if scheduler else None,
            }
            torch.save(state, setting.model_files)
            best_acc = copy.deepcopy(acc1)
        evl_end = time.time()
        log_string(log, 'One evaluate need {:.2f}s'.format(
            evl_end - evl_start))

bar.close()