from include.header import *
from collections import OrderedDict

def makeFeatureExtract_savefile():
    #model = resnet18(in_channel=3,use_dropout=True)
    first_conv = [[200, 50, 100]]
    kernel_size_list = [9, 11, 13]
    for conv_info in first_conv:
        for kernel_size in kernel_size_list:
            load_path = 'C:/Users/dongyoung/Desktop/Git/SleepEDF/model_saved/tuning/loss_fn/'

            load_file_name = 'DeepSleepNet_SleepEDF_0.0010_5_0_flod_RMS_Standard_CE.pth'
            save_file_name = 'DeepSleepNet_SleepEDF_0.0010_5_0_flod_RMS_Standard_CE_FE.pth'
            load_file = load_path + load_file_name
            state_dict = torch.load(load_file)

            new_state_dict = OrderedDict()
            for key,value in state_dict.items():
                if(key != 'fc.bias' and key != 'fc.weight'):
                    print('key : ',key)
                    #key = key[15:]
                    print('key : ', key[16:])
                    new_state_dict[key[16:]] = value
            print(new_state_dict.keys())
            save_file = load_path+save_file_name
            torch.save(new_state_dict, save_file)

makeFeatureExtract_savefile()