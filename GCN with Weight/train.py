import argparse
import time
from models import ResGCN, Net
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from utils import get_data, accuracy
import numpy as np
# --------- Sub Function ---------#

def train(epoch, max_train_acc, max_valid_acc):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss_train = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    acc_train = accuracy(output[data.train_mask], data.y[data.train_mask])
    loss_train.backward()
    optimizer.step()

    loss_val = F.nll_loss(output[data.valid_mask], data.y[data.valid_mask])
    acc_val = accuracy(output[data.valid_mask], data.y[data.valid_mask])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    Loss_Data_Log.write('Epoch: {:04d} '.format(epoch + 1) + 'loss_train: {:.4f} '.format(loss_train.item()) +
                        'acc_train: {:.4f} '.format(acc_train.item()) + 'loss_val: {:.4f} '.format(loss_val.item()) +
                        'acc_val: {:.4f} '.format(acc_val.item()) + 'time: {:.4f}s'.format(time.time() - t) + '\n')

    if acc_train.item() > max_train_acc:
        max_train_acc = acc_train.item()
        torch.save(model.state_dict(), './Experiment Model/model_features_9_20160701_train.pt')

    if acc_val.item() > max_valid_acc:
        max_valid_acc = acc_val.item()
        torch.save(model.state_dict(), './Experiment Model/model_features_9_20160701_valid.pt')

    return max_train_acc, max_valid_acc

@torch.no_grad()
def test():
    model.eval()
    output = model(data)
    loss_test = F.nll_loss(output[data.test_mask], data.y[data.test_mask])
    acc_test = accuracy(output[data.test_mask], data.y[data.test_mask])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    Loss_Data_Log.write("Test set results:" + "loss= {:.4f} ".format(loss_test.item()) + "accuracy= {:.4f}".format(acc_test.item()) + "\n")

# --------- Main Function --------#
parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true', help='Use GDC preprocessing.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.05, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 loss on parameters).')       # 默认 0.0005 可能过小
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

data = get_data("./data/file_2016_07_01/")

if args.use_gdc:                    # GDC是一种数据预处理方法，可以减少图中噪音影响，提高性能     默认为 True
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym', normalization_out='col', diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128, dim=0), exact=True)
    data = gdc(data)

use_cuda = 'cpu'
device = torch.device(use_cuda)

if use_cuda != 'cpu':
    torch.cuda.manual_seed(args.seed)

data = data.to(device)
model = ResGCN(data.x.shape[1], 2, data.edge_index, data.edge_attr).to(device)
# model = Net(data.x.shape[1], 2, False).to(device)

# optimizer = torch.optim.Adam([dict(params=model.conv1.parameters(), weight_decay=5e-4), dict(params=model.conv2.parameters(), weight_decay=0)], lr=0.01)  # Only perform weight-decay on first convolution.
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

Loss_Data_Log = open("Experiment Log/features_9_20160701.txt", "w")
Loss_Data_Log.write("实验一：model 2*3    dropout:0.5    lr:0.1" + "\n")

max_train_acc = 0
max_valid_acc = 0
t_total = time.time()
for epoch in range(args.epochs):
    max_train_acc, max_valid_acc = train(epoch, max_train_acc, max_valid_acc)
    if epoch % 5 == 0:
        test()

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
Loss_Data_Log.write("Optimization Finished!" + "\n")
Loss_Data_Log.write("Total time elapsed: {:.4f}s".format(time.time() - t_total) + "\n")

# Testing and Save Model
print("Max Train : %f" % max_train_acc, "Max Test : %f" % max_valid_acc)
test()
torch.save(model.state_dict(), './Experiment Model/model_features_8_20180401.pt')