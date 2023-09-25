import torch
from abc import abstractmethod
from numpy import inf
from torch.utils.tensorboard import SummaryWriter
from torch.utils.checkpoint import checkpoint
from utils.utils import EarlyStopping
import logging
import time
import os
import torch.distributed as dist
from pathlib import Path
join = os.path.join
from torch.nn.parallel import DistributedDataParallel as DDP
import torchmetrics


class BaseTrainer:
    """
    Base class for all trainers
    没多大可能东的设置放在此处
    """
    def __init__(self, model, optimizer, config, data_loader, 
                 valid_data_loader=None, valid_interval=5, early_stopping=15):
        self.data_loader = data_loader # 训练数据
        self.valid_data_loader = valid_data_loader  # 验证数据
        self.do_validation = self.valid_data_loader is not None  # 是否做验证
        self.valid_interval = valid_interval # 每几个epoch做一次验证
        self.config = config  # 一些超参的字典 （之后可以考虑改成arg
        self.device = "cuda"
        self.model = model.to(self.device)
        self.optimizer = optimizer 
        self.scaler = torch.cuda.amp.GradScaler()  #采用amp
        self.start_epoch = 0 # 开始的epoch，有checkpoint会更新
        self.mnt_best = 0 # 最优的验证集metric
        self.epochs = config['epochs'] # 最大训练轮次
        self.len_epoch = len(self.data_loader) # 一个epoch有多少个batch
        self.early_stopping = EarlyStopping(early_stopping) # 设置早停

        # configuration to monitor model performance and save best
        now_time = self._get_time()
        self.model_path = join(config["save_dir"], "model", now_time)
        self.logger_path = join(config["save_dir"], "log", now_time)
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.logger_path, exist_ok=True)
        self.checkpoint_dir = Path(self.model_path)
        self.logger = self._set_logger(self.logger_path)
        
        # 加载一些要用的metrics
        if "Metrics" in config:
            metrics_dict = config["Metrics"]
            self.train_metrics, self.valid_metrics = [], []
            for key in metrics_dict:
                one_metric = getattr(torchmetrics, key)
                self.train_metrics.append(one_metric(**metrics_dict[key]))
                self.valid_metrics.append(one_metric(**metrics_dict[key]))

        # 加载checkpoint
        if config["resume"] is not None:
            self._resume_checkpoint(config["resume"])
            
        # model = torch.compile(model, mode="default") #pytorch2.0全新特性，类似静态图

        # setup visualization writer instance                
        self.writer = SummaryWriter(log_dir=self.logger_path)
        
        # DDP加载
        if config["is_ddp"]:
            self.model = DDP(self.model, device_ids=[self.device])

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError
    
    def _valid_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            
            if self.do_validation and (epoch + 1) % self.valid_interval == 0:
                val_log = self._valid_epoch(epoch)
                log.update(**{'val_'+k : v for k, v in val_log.items()})
                   

            # print logged informations to the screen
            for key, value in log.items():
                if dist.get_rank() == 0:
                    if "loss" in key:
                        self.writer.add_scalars("loss", {key: value}, epoch)
                    if "dice" in key:
                        self.writer.add_scalars("dice", {key: value}, epoch)
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if "val_dice" in log.keys():
                if self.mnt_best < log["val_dice"]:
                    self.mnt_best = log["val_dice"]
                    best = True
            
            if dist.get_rank() == 0:
                self._save_checkpoint(epoch, save_best=best)
            
            self.early_stopping(log["train_loss"])
            if self.early_stopping.early_stop:
                break

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        old_file = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch-5))
        if os.path.exists(old_file):
            os.remove(old_file)
                
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            print("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        my_dic_keys = list(checkpoint['state_dict'].keys())
        for key in my_dic_keys:
            checkpoint['state_dict'][key.replace("module.", "")] = checkpoint['state_dict'].pop(key)
        self.model.eval()
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            print("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
        self.model_path = os.path.dirname(resume_path)
        self.logger_path = join(self.model_path.replace("model", "log"))
        
    def _get_time(self):
        t = time.localtime()
        now = time.strftime("%m%d_", t) + str(t.tm_hour+8)+str(t.tm_min)
        return now
    
    
    def _set_logger(self, logger_path):
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(join(logger_path, "log.txt"))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.addHandler(console)
        return logger
    
    def _train_batch_end(self, batch_loss, epoch_loss):
        self.optimizer.zero_grad()
        self.scaler.scale(batch_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        epoch_loss += batch_loss.item()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return epoch_loss
        