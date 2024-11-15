import torch


class MnistNet(torch.nn.Module):
    
    def __init__(self):
        super().__init__()

        # self.f1 = torch.nn.Linear(28*28,64)
        # self.f1_act = torch.nn.LeakyReLU()
        # self.f2 = torch.nn.Linear(64,10)
        # self.f2_act = torch.nn.Softmax(-1)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(28*28,512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512,10),
            torch.nn.Softmax(-1)
        )

    def forward(self,x):
       
        return self.fc(x)
