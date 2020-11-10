from include.header import *
from utils.function.function import *
from models.cnn.DeepSleepNet_cnn import *
from utils.function.loss_fn import *

def train_selectChannel_DeepSleepNet_cnn(train_signals_path,val_signals_path,train_list,val_list,annotations_path,
                                         save_file,logging_file,select_channel=[0],epochs=2000,learning_rate=0.001,
                                         optim='Adam',loss_function='CE',preprocessing=True,norm_methods='Standard',
                                         use_noise=False,epsilon=0.5,noise_scale=2e-6):
    # Adam optimizer param
    b1 = 0.5
    b2 = 0.999

    beta = 0.001

    check_file = open(logging_file,'w')

    best_accuracy = 0.
    best_epoch = 0

    train_dataset_len = len(train_list)
    val_dataset_len = len(val_list)

    print('Train dataset len : ',train_dataset_len)
    print('Validation dataset len : ',val_dataset_len)

    _, count = make_weights_for_balanced_classes(annotations_path, train_list, nclasses=6)
    print('label count : ',count)

    model = DeepSleepNet_classification(in_channel=len(select_channel),class_num=6)

    model.apply(weights_init) # model weight init

    cuda = torch.cuda.is_available()

    if cuda:
        print('Use Cuda')
        model = model.cuda()

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
        print('wegiths : ',weights)
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
        print('weights : ',weights)
        loss_fn = FocalLoss(gamma=2,weights=weights,no_of_classes=no_of_classes).to(device)


    if cuda:
        loss_fn = loss_fn.to(device)

    if optim == 'Adam':
        print('Optimizer : Adam')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(b1, b2))
    elif optim == 'RMS':
        print('Optimizer : RMSprop')
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optim == 'SGD':
        print('Optimizer : SGD')
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    minibatch_size = 5
    norm_square = 2
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=.5, patience=10,
                                                           min_lr=1e-6)
    cut_value = 192e-6
    if norm_methods =='OneToOne' or norm_methods =='MinMax':
        cut_value = torch.tensor(cut_value, dtype=torch.float)

    for epoch in range(epochs):
        train_dataset = suffle_dataset_list(train_list)
        count = 0

        train_total_loss = 0.0
        train_total_count = 0
        train_total_data = 0

        val_total_loss = 0.0
        val_total_count = 0
        val_total_data = 0


        start_time = time.time()
        model.train()

        output_str = 'current_lr : %f\n'%(optimizer.state_dict()['param_groups'][0]['lr'])
        sys.stdout.write(output_str)
        check_file.write(output_str)
        batch_size = 0
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

                for param in model.state_dict():
                    if param == 'feature_extract.bigCNN.conv1.weight' or param == 'feature_extract.smallCNN.conv1.weight':
                        norm += torch.norm(model.state_dict()[param], p=norm_square)

                loss = loss_fn(pred, batch_labels) + beta * norm
                # print('loss : ',loss.item())
                # loss = loss_fn(pred, batch_label)
                # acc

                _, predict = torch.max(pred, 1)

                check_count = (predict == batch_labels).sum().item()

                train_total_loss += loss.item()

                train_total_count += check_count
                train_total_data += len(batch_signals)
                loss.backward()
                optimizer.step()

                del (batch_signals)
                del (batch_labels)
                del (loss)
                del (pred)
                torch.cuda.empty_cache()
                batch_size += 1
                count = 0

        train_total_loss /= batch_size
        train_accuracy = train_total_count / train_total_data * 100

        output_str = 'train dataset : %d/%d epochs spend time : %.4f sec / total_loss : %.4f correct : %d/%d -> %.4f%%\n' \
                     % (epoch + 1, epochs, time.time() - start_time, train_total_loss,
                        train_total_count, train_total_data, train_accuracy)
        sys.stdout.write(output_str)
        check_file.write(output_str)

        # check validation dataset
        start_time = time.time()
        model.eval()
        batch_size = 0
        count = 0

        for index,filename in enumerate(val_list):
            if index % minibatch_size == 0:
                batch_signals, batch_labels = get_dataset_selectChannel(signals_path=val_signals_path,annotations_path=annotations_path,filename=filename,select_channel=select_channel,preprocessing=preprocessing,
                                                                  norm_methods=norm_methods,cut_value=cut_value)
            else:
                new_signals, new_labels = get_dataset_selectChannel(signals_path=val_signals_path,annotations_path=annotations_path,filename=filename,select_channel=select_channel,preprocessing=preprocessing,
                                                                  norm_methods=norm_methods,cut_value=cut_value)
                batch_signals = torch.cat((batch_signals, new_signals), dim=1)
                batch_labels = torch.cat((batch_labels, new_labels))
            count += 1
            if count == minibatch_size or index == len(val_list) - 1:
                batch_signals = expand_signals_torch(batch_signals,len(select_channel))
                batch_signals = batch_signals.to(device)
                batch_labels = batch_labels.to(device)


                with torch.no_grad():
                    pred = model(batch_signals)

                loss = loss_fn(pred, batch_labels)

                # acc
                _, predict = torch.max(pred, 1)
                check_count = (predict == batch_labels).sum().item()

                val_total_loss += loss.item()
                val_total_count += check_count
                val_total_data += len(batch_signals)

                # 사용하지 않는 변수 제거
                del (batch_signals)
                del (batch_labels)
                del (loss)
                del (pred)
                torch.cuda.empty_cache()
                batch_size += 1
                count = 0

        val_total_loss /= batch_size
        val_accuracy = val_total_count / val_total_data * 100

        output_str = 'val dataset : %d/%d epochs spend time : %.4f sec  / total_loss : %.4f correct : %d/%d -> %.4f%%\n' \
                     % (epoch + 1, epochs, time.time() - start_time, val_total_loss,
                        val_total_count, val_total_data, val_accuracy)
        sys.stdout.write(output_str)
        check_file.write(output_str)

        scheduler.step(float(val_total_loss))
        # scheduler.step()

        if epoch == 0:
            best_accuracy = val_accuracy
            best_epoch = epoch
            save_file = save_file
            # save_file = save_path + 'best_SleepEEGNet_CNN_channel%d.pth'%channel
            torch.save(model.state_dict(), save_file)
            stop_count = 0
        else:
            if best_accuracy < val_accuracy:
                best_accuracy = val_accuracy
                best_epoch = epoch
                save_file = save_file
                torch.save(model.state_dict(), save_file)
                stop_count = 0
            else:
                stop_count += 1
        if stop_count > 30:
            print('Early Stopping')
            break

        output_str = 'best epoch : %d/%d / val accuracy : %f%%\n' \
                     % (best_epoch + 1, epochs, best_accuracy)
        sys.stdout.write(output_str)
        print('=' * 30)

    output_str = 'best epoch : %d/%d / accuracy : %f%%\n' \
                 % (best_epoch, epochs, best_accuracy)
    sys.stdout.write(output_str)
    check_file.write(output_str)
    print('=' * 30)

    check_file.close()


def training_info(k_fold=5):

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
                        logging_file = logging_path + 'DeepSleepNet_SleepEDF_%.4f_%d_%d_flod_%s_%s_%s_twoChannel.txt'%(learning_rate,k_fold,k_num,optim,norm_methods,loss_function)
                        save_file = save_path + 'DeepSleepNet_SleepEDF_%.4f_%d_%d_flod_%s_%s_%s_twoChannel.pth'%(learning_rate,k_fold,k_num,optim,norm_methods,loss_function)

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
                        train_selectChannel_DeepSleepNet_cnn(train_signals_path=train_signals_path,val_signals_path=train_signals_path,train_list=train_list,val_list=val_list,annotations_path=annotations_path,
                                                             save_file=save_file,logging_file=logging_file,select_channel=select_channel,epochs=epochs,learning_rate=learning_rate,optim=optim,loss_function=loss_function,preprocessing=preprocessing,
                                                             norm_methods=norm_methods)

training_info(k_fold=5)