import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


# ---------------- MODEL ----------------
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, 5)

        self.fc1 = nn.Linear(24 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ---------------- LOAD MODEL ONCE ----------------
device = torch.device("cpu")

net = NeuralNet()
net.load_state_dict(torch.load("trained_net.pth", map_location=device))
net.eval()


# ---------------- CLASSES ----------------
classes = [
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck" ,"apple"
]


# ---------------- TRANSFORM (FIXED) ----------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),   # 🔥 FIXED (VERY IMPORTANT)
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ---------------- CLASSIFY FUNCTION ----------------
def classify_image(img):
    img = transform(img)
    img = img.unsqueeze(0)  # batch dimension

    with torch.no_grad():
        output = net(img)
        _, predicted = torch.max(output, 1)

    return classes[predicted.item()]
