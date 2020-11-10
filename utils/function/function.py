from include.header import *

# model conv layer weight init function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:         # Conv weight init
        torch.nn.init.xavier_uniform_(m.weight.data)

def suffle_dataset_list(dataset_list): # 데이터 셔플
    random.shuffle(dataset_list)
    return dataset_list

def search_signals_npy(dirname):
    filenames = os.listdir(dirname)
    filenames = [file for file in filenames if file.endswith(".npy")]
    return filenames

def search_correct_annotations_npy(dirname, filename):
    search_filename = filename.split('-')[0][:-2]
    file_list = os.listdir(dirname)
    filename = [file for file in file_list if search_filename in file if file.endswith("npy")]

    return filename[0]

#Standard Scaler
def data_preprocessing_torch(signals): # 하나의 데이터셋에 대한 data_preprocessing (using torch)
    signals = (signals - signals.mean(dim=1).unsqueeze(1))/signals.std(dim=1).unsqueeze(1)

    return signals

def data_preprocessing_oneToOne_torch(signals,min,max,max_value):
    signals_std = (signals + max_value) / (2*max_value)
    signals_scaled = signals_std * (max - min) + min
    return signals_scaled

def get_dataset_selectChannel_sequence(signals_path,annotations_path,filename,sequence_length=1,stride=1,select_channel=[0,1,2],use_noise=False,epsilon=0.5,noise_scale=2e-6,preprocessing=False,norm_methods='Standard',cut_value=200):
    # npy read!
    signals = np.load(signals_path + filename)

    annotations = np.load(annotations_path + search_correct_annotations_npy(annotations_path, filename))

    signals = signals[select_channel]

    signals = torch.from_numpy(signals).float()
    annotations = torch.from_numpy(annotations).long()

    total_length = len(annotations)

    epoch = 30
    sample_rate = 200

    if preprocessing:
        if norm_methods=='Standard':
            signals = data_preprocessing_torch(signals)
        elif norm_methods=='OneToOne':
            signals = torch.where(signals < -cut_value, -cut_value, signals)
            signals = torch.where(signals > cut_value, cut_value, signals)
            signals = data_preprocessing_oneToOne_torch(signals,-1,1,cut_value)
        elif norm_methods=='MinMax':
            signals = torch.where(signals < -cut_value, -cut_value, signals)
            signals = torch.where(signals > cut_value, cut_value, signals)
            signals = data_preprocessing_oneToOne_torch(signals,0,1,cut_value)

    new_signals = []
    new_label = []
    for i in range(0,total_length,stride):
        if i + sequence_length < total_length:
            new_signals.append(signals[:,i*(sample_rate*epoch):(i+sequence_length)*(sample_rate*epoch)])
            new_label.append(annotations[i:(i+sequence_length)])
        elif i + sequence_length == total_length:
            new_signals.append(signals[:, i * (sample_rate * epoch):(i + sequence_length) * (sample_rate * epoch)])
            new_label.append(annotations[i:(i + sequence_length)])
            break
        else:
            new_signals.append(signals[:, (-sequence_length) * (sample_rate * epoch):])
            new_label.append(annotations[-sequence_length:])
            break
    new_signals = torch.tensor(new_signals)
    new_label = torch.tensor(new_label)
    return new_signals, new_label

def get_dataset_selectChannel(signals_path,annotations_path,filename,select_channel=[0,1,2],use_noise=False,epsilon=0.5,noise_scale=2e-6,preprocessing=False,norm_methods='Standard',cut_value=200):
    signals = np.load(signals_path+filename)

    annotations = np.load(annotations_path+search_correct_annotations_npy(annotations_path,filename))

    signals = signals[select_channel]

    signals = torch.from_numpy(signals).float()
    annotations = torch.from_numpy(annotations).long()

    if preprocessing:
        if norm_methods=='Standard':
            signals = data_preprocessing_torch(signals)
        elif norm_methods=='OneToOne':
            signals = torch.where(signals < -cut_value, -cut_value, signals)
            signals = torch.where(signals > cut_value, cut_value, signals)
            signals = data_preprocessing_oneToOne_torch(signals,-1,1,cut_value)
        elif norm_methods=='MinMax':
            signals = torch.where(signals < -cut_value, -cut_value, signals)
            signals = torch.where(signals > cut_value, cut_value, signals)
            signals = data_preprocessing_oneToOne_torch(signals,0,1,cut_value)

    return signals,annotations

def expand_signals_torch(signals,channel_len,sample_rate=100,epoch_sec=30):
    signals = signals.unsqueeze(0)
    #print(signals.shape)
    signals = signals.transpose(1,2)
    #print(batch_signals.shape)
    signals = signals.view(-1,sample_rate*epoch_sec,channel_len)
    #print(batch_signals.shape)
    signals = signals.transpose(1,2)
    return signals