import os
import numpy as np
import re

def fill_missing_elements(partial_elements, partial_values):
    full_elements = list(range(1, 65))  # Elements from 1 to 64
    filled_elements = full_elements.copy()
    filled_values = [0] * len(full_elements)

    # Fill the known values
    for idx, element in enumerate(partial_elements):
        index = full_elements.index(element)
        filled_values[index] = partial_values[idx]

    # Interpolate the missing values
    for idx, value in enumerate(filled_values):
        if value == 0:
            left_index = idx - 1
            right_index = idx + 1

            while left_index >= 0 and filled_values[left_index] == 0:
                left_index -= 1

            while right_index < len(filled_values) and filled_values[right_index] == 0:
                right_index += 1

            if left_index >= 0 and right_index < len(filled_values):
                filled_values[idx] = (filled_values[left_index] + filled_values[right_index]) / 2
            elif left_index >= 0:
                filled_values[idx] = filled_values[left_index]
            elif right_index < len(filled_values):
                filled_values[idx] = filled_values[right_index]

    return filled_elements, filled_values

def extract_info_from_filename(filename):
    pattern = re.compile(r'group(\d+)_src(\d+)_(\w)\.fbp')
    match = pattern.match(filename)

    if match:
        group_num = int(match.group(1))
        src_num = int(match.group(2))
        flag = match.group(3) == 'P'
    else:
        print(f"Invalid filename format: {filename}")
        return None, None, None

    return group_num, src_num, flag

def process_file_to_get_t(filename):
    with open(filename) as f:
        lines = f.readlines()
    lines.remove(lines[0])  # remove the header line

    trace_numbers = [int(line.split()[0]) for line in lines]
    trace_times = [float(line.split()[2]) for line in lines]

    # Remove elements before the first occurrence of 1 or the next smallest number
    if 1 in trace_numbers:
        start_index = trace_numbers.index(1)
    else:
        start_index = trace_numbers.index(min(trace_numbers))
        
    trace_numbers = trace_numbers[start_index:] + trace_numbers[:start_index]
    trace_times = trace_times[start_index:] + trace_times[:start_index]

    if len(trace_numbers) != 64:
        filled_elements, filled_values = fill_missing_elements(trace_numbers, trace_times)
    else:
        filled_elements, filled_values = trace_numbers, trace_times

    t = np.zeros((1000, 64))
    for i in range(len(filled_elements)):
        t[int(filled_values[i]), filled_elements[i] - 1] = 1

    return t

def process_file_to_get_x(group_num, src_num, base_path='./Data/'):
    channel_names = ['_vx', '_vy', '_vz']
    channels = []
    
    for channel_name in channel_names:
        file_path = f"{base_path}group{group_num}_src{src_num}{channel_name}.csv"
        if not os.path.isfile(file_path):
            print(f"File not found: {file_path}")
            return None
        
        data = np.genfromtxt(file_path, delimiter=',')
        channels.append(data)
    
    x_sample = np.stack(channels, axis=0)
    return x_sample


# Initialize Y as an empty dictionary
Y = {}
X = {}

filelist = os.listdir('./Pick')
for filename in filelist:
    group_num, src_num, is_P = extract_info_from_filename(filename)
    if group_num is None:
        continue
    t = process_file_to_get_t('./Pick/'+filename)
    x_sample = process_file_to_get_x(group_num, src_num)


    # Combine the group_num and src_num as a tuple to use as a key for the Y dictionary
    key = (group_num, src_num)
    if key not in Y:
        Y[key] = np.zeros((2, 1000, 64))
        X[key] = np.zeros((3, 1000, 64))

    # Assign the t array to the appropriate position in the Y dictionary
    Y[key][0 if is_P else 1] = t
    X[key] = x_sample

# Convert the Y dictionary to a NumPy array with the desired shape (number_of_samples, 2, 1000, 64)
Y = np.stack(list(Y.values()), axis=0)
X = np.stack(list(X.values()), axis=0)

# np.save('X.npy', X)
# np.save('Y.npy', Y)

# ------------Pre-process the raw data as training set----------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt

device = torch.device('mps')
import numpy as np
X = np.load('X.npy')
Y = np.load('Y.npy')
# select the arrival and make it a band with 15 pixels length
for i in range(len(Y)):
    for j in range(64):
        ipwave = np.argmax(Y[i,0,:,j])
        Y[i,0,ipwave-7:ipwave+7,j] = 1
        iswave = np.argmax(Y[i,1,:,j])
        Y[i,1,iswave-7:iswave+7,j] = 1

class MyDataset(Dataset):
    def __init__(self, X, Y):
        from torchvision import transforms
        self.X = X
        self.Y = Y
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]

        x_tensor = torch.tensor(x).float()  # Convert to float
        y_tensor = torch.tensor(y).float()  # Convert to float

        return x_tensor, y_tensor

X1 = X[:,:,:-1,:]
Y1 = Y
batch_size = 4

X_train, X_val, Y_train, Y_val = train_test_split(X1,
                                                  Y1, 
                                                  test_size=0.1, random_state=37)

# Create PyTorch datasets and data loaders for training and validation
train_dataset = MyDataset(X_train, Y_train)
val_dataset = MyDataset(X_val, Y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Instantiate the U-Net model and optimizer
# model = UNet(3,2)  # Assuming UNet is your U-Net model class
#model = UNetFB()
optimizer = optim.Adam(model.parameters(), lr=0.00001)
criterion = nn.BCEWithLogitsLoss()

# Training loop
num_epochs = 13
running_loss = 0.0
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    model.train()
    print(f"Epoch {epoch + 1}/{num_epochs}")

    i = 0
    running_loss = 0.0
    for batch_X, batch_Y in train_loader:
        i += 1
        # batch_X has dimensions (batch_size, 3, 1000, 64)
        # batch_Y has dimensions (batch_size, 2, 1000, 64)

        # Forward pass
        output = model(batch_X)  # output has dimensions (batch_size, 2, 1000, 64)

        # Calculate the loss
        train_loss = criterion(output, batch_Y)

        # Backward pass
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step() 
        
        # print statistics
        running_loss += train_loss.item()
        if i % 20 == 19:    # print every 20 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/i:.3f}')
            # running_loss = 0.0
    train_losses.append(running_loss)
    print(f"Loss: {running_loss}")
    
    j = 0
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch_X_val, batch_Y_val in val_loader:
            j += 1
            output = model(batch_X_val)
            valloss = criterion(output, batch_Y_val)
            val_loss += valloss.item()
    val_losses.append(val_loss)
    print(f"Validation Loss: {val_loss}")


torch.save(model, 'model_13epoch_segTwoBand.pth')

def output_P_and_S(X):
    import pdb; 
    '''
    X comes in np.ndarray in [40,3,1000,64]. 43 synthetic seismogram in three component. Each is [1000, 64]
    Each forward modeling come with one source. So actually this first number can never be 43.
    It has to be strictlly ONE!!!
    
    Select num_peaks candidates. These candidates will be used later to form a connected line.
    return index_p, index_s, icol_pstart, icol_sstart
    index_p and index_s in size (num_peaks, 64)
    icol_pstart, icol_sstart are index of the most reliable event, which can be used as the start point
    to be connected and form a continous line of arrival.
    '''
    import numpy as np
    from scipy.signal import find_peaks
    from scipy.special import softmax
    from torch.utils.data import DataLoader, Dataset, TensorDataset
    import torch
    model = torch.load('model_13epoch_segTwoBand.pth')
    model.eval()
    winlen = 30
    num_peaks = 5
        
    X1 = X[:,:,:-1,:] # in case of input is 1001 instead of 1000.
    Y1 = np.zeros(X1.shape)
    test_dataset = MyDataset(X1.transpose(0, 3, 2, 1), Y1.transpose(0, 3, 2, 1))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    index_p = []

    index_s = []
    iseismogram = 0
    icol_pstart = 0
    icol_sstart = 0
    prob_p_min  = 99
    prob_s_max  = -99
    for batch_test_X, batch_test_Y in test_loader:
        sm = softmax(model(batch_test_X).detach().numpy(), axis=1)

        for i in range(sm[0,0].transpose().shape[1]):

            trace = sm[0,0].transpose()[:,i]
            shift= np.pad(trace,(int(winlen/2),int(winlen/2)),mode='edge')
            shift_forward = np.roll(shift,int(winlen/2))
            shift_backward = np.roll(shift,-int(winlen/2))
            diff = shift_backward - shift_forward
            peaks,_ = find_peaks(-diff, width= 10)
            sort_index = np.argsort(diff[peaks])
            sort_peaks = peaks[sort_index]
            ipeaks = sort_peaks[:num_desire]
            index_p.append(ipeaks)
            if diff[sort_peaks[0]] < prob_p_min:
                prob_p_min = diff[sort_peaks[0]]
                icol_pstart = i
            
            trace = sm[0,2].transpose()[:,i]
            shift= np.pad(trace,(int(winlen/2),int(winlen/2)),mode='edge')
            shift_forward = np.roll(shift,int(winlen/2))
            shift_backward = np.roll(shift,-int(winlen/2))
            diff = shift_backward - shift_forward
            peaks,_ = find_peaks(diff, width= 10)
            sort_index = np.argsort(diff[peaks])
            sort_peaks = peaks[sort_index[::-1]]
            ipeaks = sort_peaks[:num_desire]
            index_s.append(ipeaks)
            if diff[sort_peaks[0]] > prob_s_max:
                prob_s_max = diff[sort_peaks[0]]
                icol_sstart = i
            
            
        iseismogram += 1

    index_p = fill_and_stack_columns(index_p)
    index_s = fill_and_stack_columns(index_s)
    return index_p, index_s, icol_pstart, icol_sstart



def fill_and_stack_columns(columns_list):
    '''
    input a list of candidates of P- or S- arrival, they might has different number of candidates.
    This function will fill the unknown value in these list with zero and make them a same length.
    '''
    import numpy as np
    # Find the maximum column size
    max_column_size = max([column.size for column in columns_list])

    # Pad smaller columns with zeros and stack them horizontally
    padded_columns = []
    for column in columns_list:
        padded_column = np.pad(column, (0, max_column_size - column.size), 'constant', constant_values=(0))
        padded_columns.append(padded_column)

    array_with_padded_columns = np.column_stack(padded_columns)

    return array_with_padded_columns

def create_continuous_line(input_array, start_column):
    '''
    Select a start_column, given by the largest difference, which means rapid probablity increase/decrease
    and thus higher reliable.
    '''
    continuous_line = [None] * input_array.shape[1]

    # Create a mask to identify valid points (non-zero values)
    mask = input_array != 0

    # Search to the left of the start_column
    prev_y = None
    for x in range(start_column, -1, -1):
        col = input_array[:, x]
        valid_points = col[mask[:, x]]

        if prev_y is None:
            selected_y = valid_points[0]
        else:
            selected_y = valid_points[np.argmin(np.abs(valid_points - prev_y))]

        continuous_line[x] = selected_y
        prev_y = selected_y

    # Search to the right of the start_column
    prev_y = continuous_line[start_column]
    for x in range(start_column + 1, input_array.shape[1]):
        col = input_array[:, x]
        valid_points = col[mask[:, x]]

        selected_y = valid_points[np.argmin(np.abs(valid_points - prev_y))]

        continuous_line[x] = selected_y
        prev_y = selected_y

    return continuous_line
            