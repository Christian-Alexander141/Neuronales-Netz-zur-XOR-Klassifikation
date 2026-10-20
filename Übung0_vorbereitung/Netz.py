import torch
import torch.nn as nn
import torch.optim as optim

# XOR-Daten
X = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

# Einfaches neuronales Netz mit 1 versteckten Schicht
class XORNet(nn.Module):
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # Eingabe -> versteckte Schicht (2 Neuronen)
        self.fc2 = nn.Linear(2, 1)  # Versteckte Schicht -> Ausgabe

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # Aktivierungsfunktion: Sigmoid
        x = torch.sigmoid(self.fc2(x))
        return x

# Modell, Verlustfunktion und Optimierer
model = XORNet()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training
for epoch in range(10_000):
    output = model(X)
    loss = criterion(output, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

# Ergebnisse
print("\nTest:")
with torch.no_grad():
    predictions = model(X)
    print(torch.round(predictions))  # Rundet auf 0 oder 1
