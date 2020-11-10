from include.header import *
from utils.function.function import *
from utils.function.loss_fn import *
from models.rnn.DeepSleepNet_lstm import *
from models.cnn.DeepSleepNet_cnn import *
from models.DeepSleepNet import *

SOD_token = 5
EOS_token = 5

def train_model_lstm_onechannel(train_signals_path,val_signals_path,train_list,val_list,annotations_path,load_FE_filename,
                                         save_file,logging_file,select_channel=[0],epochs=2000,learning_rate=0.001,
                                         optim='Adam',loss_function='CE',preprocessing=True,norm_methods='Standard',
                                         use_noise=False,epsilon=0.5,noise_scale=2e-6):
    # Adam optimizer param
    b1 = 0.5
    b2 = 0.999
    class_num = 6
    beta = 0.001

    check_file = open(logging_file, 'w')

    best_accuracy = 0.
    best_epoch = 0

    train_dataset_len = len(train_list)
    val_dataset_len = len(val_list)

    print('Train dataset len : ', train_dataset_len)
    print('Validation dataset len : ', val_dataset_len)

    _, count = make_weights_for_balanced_classes(annotations_path, train_list, nclasses=class_num)
    print('label count : ', count)

    hidden_dim = 512

    FeatureExtract = DeepSleepNet_pretrained(in_channel=1,sequence_length=1,class_num=class_num)

    Classification = BiLSTMClassification(input_size=3456,class_num=class_num,hidden_dim=hidden_dim,num_layers=1)

    FeatureExtract.apply(weights_init)
    Classification.apply(weights_init)  # model weight init

    cuda = torch.cuda.is_available()

    if cuda:
        print('Use Cuda')
        FeatureExtract = FeatureExtract.cuda()
        Classification = Classification.cuda()


    load_fe_file = load_FE_filename

    FeatureExtract.featureExtract.load_state_dict(torch.load(load_fe_file))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print('Multi GPU Activation')
    else:
        print('One GPU Process!!!')

    if loss_function == 'CE':
        loss_fn = nn.CrossEntropyLoss().to(device)
    elif loss_function == 'CEW':
        samples_per_cls = count / np.sum(count)
        no_of_classes = 6
        if no_of_classes == 6:
            effective_num = 1.0 - np.power(beta, samples_per_cls[:5])
            weights = (1.0 - beta) / np.array(effective_num)
            weights = np.append(weights, 0.)
        else:
            effective_num = 1.0 - np.power(beta, samples_per_cls)
            weights = (1.0 - beta) / np.array(effective_num)

        weights = weights / np.sum(weights) * no_of_classes

        weights = torch.tensor(weights).float()
        weights = weights.to(device)
        print('wegiths : ', weights)
        loss_fn = nn.CrossEntropyLoss(weight=weights).to(device)
    elif loss_function == 'FL':
        loss_fn = FocalLoss(gamma=2).to(device)
    elif loss_function == 'CBL':
        samples_per_cls = count / np.sum(count)
        no_of_classes = 6
        if no_of_classes == 6:
            effective_num = 1.0 - np.power(beta, samples_per_cls[:5])
            weights = (1.0 - beta) / np.array(effective_num)
            weights = np.append(weights, 0.)
        else:
            effective_num = 1.0 - np.power(beta, samples_per_cls)
            weights = (1.0 - beta) / np.array(effective_num)

        weights = weights / np.sum(weights) * no_of_classes

        weights = torch.tensor(weights).float()
        weights = weights.to(device)
        print('weights : ', weights)
        loss_fn = FocalLoss(gamma=2, weights=weights, no_of_classes=no_of_classes).to(device)

    if cuda:
        loss_fn = loss_fn.to(device)

    if optim == 'Adam':
        print('Optimizer : Adam')
        optimizer = torch.optim.Adam([{'params': FeatureExtract.parameters(), 'lr': 0},
                                     {'params': Classification.parameters()}], lr=learning_rate)
    elif optim == 'RMS':
        print('Optimizer : RMSprop')
        # optimizer = torch.optim.RMSprop(model.parameters(),lr=0)
        optimizer = torch.optim.RMSprop([{'params': FeatureExtract.parameters(), 'lr': 0},
                                     {'params': Classification.parameters()}], lr=learning_rate)
    elif optim == 'SGD':
        print('Optimizer : SGD')
        optimizer = torch.optim.SGD([{'params': FeatureExtract.parameters(), 'lr': 0},
                                     {'params': Classification.parameters()}], lr=learning_rate, momentum=0.9)

    minibatch_size = 5
    norm_square = 2

    cut_value = 192e-6
    if norm_methods == 'OneToOne' or norm_methods == 'MinMax':
        cut_value = torch.tensor(cut_value, dtype=torch.float)



    best_accuracy = 0.
    stop_count = 0
    for epoch in range(epochs):
        train_dataset = suffle_dataset_list(train_list) # 매 epoch마다 train_dataset shuffle !
        count = 0  # check batch
        train_total_loss = 0.0

        start_time = time.time()
        FeatureExtract.eval()
        Classification.train()


        output_str = 'FeatureExtract_lr : %f\nclassification_lr : %f\n'%(optimizer.state_dict()['param_groups'][0]['lr'],optimizer.state_dict()['param_groups'][1]['lr'])
        # output_str = 'classification_lr : %f\n' % (
        #     optimizer.state_dict()['param_groups'][0]['lr'])
        sys.stdout.write(output_str)
        check_file.write(output_str)

        for index,filename in enumerate(train_dataset):
            # print('index : ',index)

            if index % minibatch_size == 0:
                batch_signals, batch_labels = get_dataset_selectChannel(signals_path=train_signals_path,annotations_path=annotations_path,filename=filename,select_channel=select_channel,preprocessing=preprocessing,
                                                                  norm_methods=norm_methods,cut_value=cut_value)
            else:
                new_signals, new_labels = get_dataset_selectChannel(signals_path=train_signals_path,annotations_path=annotations_path,filename=filename,select_channel=select_channel,preprocessing=preprocessing,
                                                                  norm_methods=norm_methods,cut_value=cut_value)

                batch_signals = torch.cat((batch_signals, new_signals), dim=1)
                batch_labels = torch.cat((batch_labels, new_labels))
            count += 1
            if count == minibatch_size or index == len(train_dataset) - 1:  # batch 학습 시작!

                batch_signals = expand_signals_torch(batch_signals,len(select_channel))
                batch_signals = batch_signals.to(device)
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad()
                # print(batch_signal.shape)
                # print(batch_signal)
                pred = model(batch_signals)
                norm = 0
        train_total_count = 0
        train_total_data = 0

        exit(1)

def training_rnn_info(k_fold=5):

    train_signals_path = 'C:/dataset/SleepEDF/origin_npy/3channel/remove_wake/train/'

    annotations_path = 'C:/dataset/SleepEDF/annotations/remove_wake/'

    save_path = 'C:/Users/dongyoung/Desktop/Git/SleepEDF/model_saved/tuning/loss_fn/'
    logging_path = 'C:/Users/dongyoung/Desktop/Git/SleepEDF/log/tuning/loss_fn/'

    os.makedirs(save_path,exist_ok=True)
    os.makedirs(logging_path,exist_ok=True)
    epochs = 2000
    learning_rate_list = [0.001]

    optim_list = ['RMS','Adam']
    loss_function_list = ['CE']
    preprocessing = True
    dataset_list = os.listdir(train_signals_path)
    norm_methods_list = ['Standard']
    #for k_num in range(0, k_fold):
    for k_num in range(0,1):
        for learning_rate in learning_rate_list:
            for optim in optim_list:
                for norm_methods in norm_methods_list:
                    for loss_function in loss_function_list:
                        print('learning_rate : %d' % learning_rate)
                        logging_file = logging_path + 'DeepSleepNet_SleepEDF_%.4f_%d_%d_flod_%s_%s_%s_lstm.txt'%(learning_rate,k_fold,k_num,optim,norm_methods,loss_function)
                        save_file = save_path + 'DeepSleepNet_SleepEDF_%.4f_%d_%d_flod_%s_%s_%s_lstm.pth'%(learning_rate,k_fold,k_num,optim,norm_methods,loss_function)
                        load_FE_filename = save_path + 'DeepSleepNet_SleepEDF_%.4f_%d_%d_flod_%s_%s_%s_FE.pth'%(learning_rate,k_fold,k_num,optim,norm_methods,loss_function)
                        train_list = []
                        val_list = []
                        for index, filename in enumerate(dataset_list):
                            if (k_num+index) % k_fold == 0:
                                val_list.append(filename)
                            else:
                                train_list.append(filename)

                        print('%d fold train len : %d / val len : %d_'%(k_num,len(train_list),len(val_list)))
                        select_channel = [0,1]
                        print('trainset ')
                        print(train_list)
                        print('valset')
                        print(val_list)
                        train_model_lstm_onechannel(train_signals_path=train_signals_path,val_signals_path=train_signals_path,train_list=train_list,val_list=val_list,annotations_path=annotations_path,load_FE_filename=load_FE_filename,
                                                             save_file=save_file,logging_file=logging_file,select_channel=select_channel,epochs=epochs,learning_rate=learning_rate,optim=optim,loss_function=loss_function,preprocessing=preprocessing,
                                                             norm_methods=norm_methods)

training_rnn_info(k_fold=5)