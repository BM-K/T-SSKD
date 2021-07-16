import os
import torch
import logging
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)
writer = SummaryWriter()


class Metric():

    def __init__(self, args):
        self.args = args

    def cal_acc(self, yhat, y):
        with torch.no_grad():
            yhat = yhat.max(dim=-1)[1]  # [0]: max value, [1]: index of max value
            acc = (yhat == y).float().mean()

        return acc

    def cal_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

        return elapsed_mins, elapsed_secs

    def draw_graph(self, cp):
        writer.add_scalars('loss_graph', {'train': cp['tl'], 'valid': cp['vl']}, cp['ep'])
        writer.add_scalars('acc_graph', {'train': cp['tma'], 'valid': cp['vma']}, cp['ep'])

    def performance_check(self, cp):
        print(f'\t==Epoch: {cp["ep"] + 1:02} | Epoch Time: {cp["epm"]}m {cp["eps"]}s==')
        print(f'\t==Train Loss: {cp["tl"]:.4f} | Train acc: {cp["tma"]:.4f}==')
        print(f'\t==Valid Loss: {cp["vl"]:.4f} | Valid acc: {cp["vma"]:.4f}==')

    def print_size_of_model(self, model):
        torch.save(model.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p") / 1e6)
        os.remove('temp.p')

    def move2device(self, sample, device):
        if len(sample) == 0:
            return {}

        def _move_to_device(maybe_tensor, device):
            if torch.is_tensor(maybe_tensor):
                return maybe_tensor.to(device)
            elif isinstance(maybe_tensor, dict):
                return {
                    key: _move_to_device(value, device)
                    for key, value in maybe_tensor.items()
                    }
            elif isinstance(maybe_tensor, list):
                return [_move_to_device(x, device) for x in maybe_tensor]
            elif isinstance(maybe_tensor, tuple):
                return [_move_to_device(x, device) for x in maybe_tensor]
            else:
                return maybe_tensor

        return _move_to_device(sample, device)

    def save_model(self, config, cp, pco):
        if not os.path.exists(config['args'].path_to_save):
            os.makedirs(config['args'].path_to_save)

        sorted_path = config['args'].path_to_save + '/checkpoint-epoch-{}-loss-{}.pt'.format(str(cp['ep'] + 1),
                                                                                             round(cp['vl'], 4))
        if cp['vl'] < pco['best_valid_loss']:
            pco['early_stop_patient'] = 0
            pco['best_valid_loss'] = cp['vl']
            torch.save(config['model'].state_dict(), sorted_path)
            print(f'\n\t## SAVE valid_loss: {cp["vl"]:.4f} | valid acc: {cp["vma"]:.4f} ##')
        else:
            pco['early_stop_patient'] += 1
            if pco['early_stop_patient'] == config['args'].patient:
                pco['early_stop'] = True
                writer.close()

        self.draw_graph(cp)
        self.performance_check(cp)