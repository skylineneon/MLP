from model import MnistNet
from torch.utils.data import DataLoader
from data import MnistDataset
from torch.optim import Adam
import torch

class Trainer:


    def __init__(self):
        self._net = MnistNet().cuda() #创建网络模型

        self._train_dataset = MnistDataset("datas/train")
        self._train_dataloader = DataLoader(self._train_dataset,batch_size=5000,shuffle=True)
        
        self._test_dataset = MnistDataset("datas/test")
        self._test_dataloader = DataLoader(self._test_dataset,batch_size=10000,shuffle=True)

        self._opt = Adam(self._net.parameters(),lr=0.0001,betas=(0.85,0.95))


    def __call__(self):
        
        for _epoch in range(10000000000000):
            
            #训练
            self._net.train()
            _loss_sum = 0.
            for _i,(_data,_target) in enumerate(self._train_dataloader): 
                _data = _data.cuda()
                _target = _target.cuda()
                _y = self._net(_data)

                _loss = torch.mean((_y - _target)**2)
                self._opt.zero_grad()
                _loss.backward()
                self._opt.step()

                _loss_sum += _loss.detach().cpu().item()
            
            print("loss",_loss_sum/len(self._train_dataloader))
        
            #测试
            self._net.eval()
            
            _acc_sum = 0
            for _i,(_data,_target) in enumerate(self._test_dataloader):
                _data = _data.cuda()
                _target = _target.cuda()
                _y = self._net(_data)
                _acc_sum += (_y.argmax(-1) == _target.argmax(-1)).sum()
            
            print(_acc_sum/len(self._test_dataset))


if __name__ == "__main__":
    train = Trainer()
    train()