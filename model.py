import torch.nn as nn

#Создаём свою модель нейронной сети из библиотеки PyTorch 
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # Создаём 4 слоя
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, num_classes)
        
        # Создаём функцию активации, она будет применятся к выходным данным для введения нелинейности в модель
        self.relu = nn.ReLU()
    
    # Метод forward последовательно активирует все слои
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        return out
