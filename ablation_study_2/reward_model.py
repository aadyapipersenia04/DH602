import os
import numpy as np
import json
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#from torchsummary import summary
from torchinfo import summary

print(len(os.listdir("/home/varad/aadya/new_annotations")))
print(len(os.listdir("/home/varad/aadya/generated_samples")))
# print(len(os.listdir("/home/varad/reward_model/new_annotations")))
print(len(os.listdir('/home/varad/og')))
# Function to extract features from annotations
def extract_features(annotation_file, labels):
    with open(annotation_file, 'r') as json_file:
        load_file = json.load(json_file)
        features = np.zeros(len(labels), dtype=int)
        if len(load_file['result']) != 0:
            choices = load_file['result'][0]['value']['choices']
            for i, label in enumerate(labels):
                if label in choices:
                    features[i] = 1
    return features

image_directory = "/home/varad/aadya/generated_samples"

annotation_directory = "/home/varad/aadya/new_annotations"

labels = ['10 - 15 microns', 'Purple to abundant pink cytoplasm with many fine, lilac granules',
          'Lobulated (segmented) nucleus with 2 - 5 lobes connected by a thin filament of chromatin',
          'Nucleus: Cytoplasmic ratio is 33% approx', 'Dense chromatin with NO nucleolus', 'Immature form of neutrophil']

X_gen = []  # List to store images
Y_gen = []  # List to store features


# Synthetic (generated) Images 
for image_filename in os.listdir(image_directory):
    if image_filename.endswith('.png'):
        image_path = os.path.join(image_directory, image_filename)
        annotation_filename = image_filename[:-4] + '.json'
        annotation_path = os.path.join(annotation_directory, annotation_filename)
        
        if os.path.exists(annotation_path):
            features = extract_features(annotation_path, labels)
            if features[-1] != 1: # removing immature form of neutrophils from synthetic dataset
                image = Image.open(image_path)
                X_gen.append(np.array(image))
                if np.sum(features) == 5:
                    features = 1
                else:
                    features = 0 
                Y_gen.append(features)  
# exit()
X_gen = np.array(X_gen)
Y_gen = np.array(Y_gen)
print("Shape of synthetic imgs", X_gen.shape)
print("Shape of synthetic imgs label", Y_gen.shape)

# exit()
# Real Images
X_real = []
Y_real = []
for image_filename in os.listdir('/home/varad/og'):
    if image_filename.endswith('.png'):
        image_path = os.path.join('/home/varad/og', image_filename)
        
        # Load the image
        image = Image.open(image_path)
        # Convert the image to numpy array and append to X
        image_np = np.array(image)
        X_real.append(image_np)
        label = np.ones(1)
        Y_real.append(label)
        

X_real = np.array(X_real)
Y_real = np.squeeze(np.array(Y_real))
print(X_real.shape)
print(Y_real.shape)
# exit()
# print(X.shape) # (1024,64,64,3)
# print(Y.shape) # (1024,6)


# Concatenate the lists (synthetic + generated)
X = np.concatenate((X_gen, X_real), axis=0)
Y = np.concatenate((Y_gen, Y_real), axis=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else exit("GPU not available"))
print("Device:", device)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).permute(0,3,1,2).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).permute(0,3,1,2).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

print(X_train_tensor.shape)
# exit()

class MyResnext50(nn.Module):
    def __init__(self, my_pretrained_model):
        super(MyResnext50, self).__init__()
        self.pretrained = my_pretrained_model
        self.pretrained.fc = nn.Identity()
        self.fc = nn.Sequential(nn.Linear(2048, 100),
                                # nn.ReLU(),
                                # nn.Linear(300,100),
                                # nn.ReLU(),
                                # nn.Linear(100, 32),
                                nn.ReLU(),
                                nn.Linear(100, 1),
                                nn.Sigmoid())

    def forward(self, x):
        x = self.pretrained(x) # get the embedding (size 1000) of the image from the resnext50 model
        output = self.fc(x)
        return output

def configure_optimizers(model, lr=0.00001, weight_decay=0.0005, lr_decay_every_x_epochs=5):
    #optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    # print('LR decay every epoch = '+str(lr_decay_every_x_epochs))
    #scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=lr_decay_every_x_epochs, gamma=gamma)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=lr_decay_every_x_epochs)
    return optimizer, scheduler

# Initialize your model and move it to the appropriate device
resnext50_pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d')
model = MyResnext50(my_pretrained_model=resnext50_pretrained)
model=model.to(device)
summary(model=model, input_size=(16, 3, 224, 224))

# Define your loss function and optimizer
criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)

# print("==> Configuring optimizer.")
# optimizer, lr_scheduler = configure_optimizers(model)
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# Train your model
num_epochs = 30
batch_size = 8
num_batches = len(X_train_tensor) // batch_size
train_loss_arr=[]
test_loss_arr=[]
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size

        # Get the current batch
        inputs = X_train_tensor[start_idx:end_idx]
        labels = y_train_tensor[start_idx:end_idx]

        optimizer.zero_grad()
        outputs = torch.squeeze(model(inputs))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Calculating training accuracy
        pred_np = outputs.detach().cpu().numpy()
        pred_np[pred_np>0.5] = 1
        pred_np = pred_np.astype(int)
        total_train += labels.size(0)
        labels = labels.detach().cpu().numpy()
        labels = labels.astype(int)
        rows_equal = pred_np == labels
        total_equal_rows = np.sum(rows_equal)
        correct_train += total_equal_rows
        

    train_loss = running_loss / len(X_train_tensor)
    train_accuracy = correct_train / total_train
    train_loss_arr.append(train_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
    # lr_scheduler.step()

    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        outputs =  torch.squeeze(model(X_test_tensor))
        test_loss = criterion(outputs, y_test_tensor).item()
        total_test = len(y_test_tensor)

        # Calculate test accuracy
        pred_np = outputs.detach().cpu().numpy()
        pred_np[pred_np>0.5] = 1
        pred_np = pred_np.astype(int)
        labels = y_test_tensor.detach().cpu().numpy()
        labels = labels.astype(int)
        rows_equal = pred_np == labels
        total_equal_rows = np.sum(rows_equal)
        correct_test += total_equal_rows
        
    test_loss_arr.append(test_loss)
    test_accuracy = correct_test / total_test

    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

torch.save(model.state_dict(), 'results/saved_models/best_model_ablation_3.pth')

# Plot the test losses
plt.plot(test_loss_arr, label='Test Losses')
plt.plot(train_loss_arr, label='Train Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Losses Over Epochs')
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig(f'results/losses_plot_resnext_{num_epochs}eps_1layerFc_ablation_study_3.png')
