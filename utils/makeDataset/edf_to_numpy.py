from include.header import *

train_list = ['SC4001EC-PSG.npy', 'SC4002EC-PSG.npy', 'SC4011EH-Hypnogram.npy', 'SC4012EC-Hypnogram.npy', 'SC4021EH-Hypnogram.npy', 'SC4031EC-Hypnogram.npy', 'SC4032EP-Hypnogram.npy', 'SC4041EC-Hypnogram.npy', 'SC4042EC-Hypnogram.npy', 'SC4061EC-Hypnogram.npy', 'SC4062EC-Hypnogram.npy', 'SC4071EC-Hypnogram.npy', 'SC4072EH-Hypnogram.npy', 'SC4081EC-Hypnogram.npy', 'SC4082EP-Hypnogram.npy', 'SC4091EC-Hypnogram.npy', 'SC4092EC-Hypnogram.npy', 'SC4101EC-Hypnogram.npy', 'SC4102EC-Hypnogram.npy', 'SC4112EC-Hypnogram.npy', 'SC4121EC-Hypnogram.npy', 'SC4122EV-Hypnogram.npy', 'SC4131EC-Hypnogram.npy', 'SC4141EU-Hypnogram.npy', 'SC4142EU-Hypnogram.npy', 'SC4151EC-Hypnogram.npy', 'SC4152EC-Hypnogram.npy', 'SC4161EC-Hypnogram.npy', 'SC4162EC-Hypnogram.npy', 'SC4171EU-Hypnogram.npy', 'SC4172EC-Hypnogram.npy', 'SC4182EC-Hypnogram.npy', 'SC4192EV-Hypnogram.npy', 'SC4201EC-Hypnogram.npy', 'SC4202EC-Hypnogram.npy', 'SC4211EC-Hypnogram.npy', 'SC4212EC-Hypnogram.npy', 'SC4221EJ-Hypnogram.npy', 'SC4231EJ-Hypnogram.npy', 'SC4232EV-Hypnogram.npy', 'SC4242EA-Hypnogram.npy', 'SC4251EP-Hypnogram.npy', 'SC4271FC-Hypnogram.npy', 'SC4272FM-Hypnogram.npy', 'SC4281GC-Hypnogram.npy', 'SC4282GC-Hypnogram.npy', 'SC4291GA-Hypnogram.npy', 'SC4292GC-Hypnogram.npy', 'SC4301EC-Hypnogram.npy', 'SC4302EV-Hypnogram.npy', 'SC4311EC-Hypnogram.npy', 'SC4312EM-Hypnogram.npy', 'SC4321EC-Hypnogram.npy', 'SC4322EC-Hypnogram.npy', 'SC4331FV-Hypnogram.npy', 'SC4332FC-Hypnogram.npy', 'SC4351FA-Hypnogram.npy', 'SC4352FV-Hypnogram.npy', 'SC4362FC-Hypnogram.npy', 'SC4371FA-Hypnogram.npy', 'SC4382FW-Hypnogram.npy', 'SC4401EC-Hypnogram.npy', 'SC4402EW-Hypnogram.npy', 'SC4411EJ-Hypnogram.npy', 'SC4412EM-Hypnogram.npy', 'SC4421EA-Hypnogram.npy', 'SC4422EA-Hypnogram.npy', 'SC4451FY-Hypnogram.npy', 'SC4461FA-Hypnogram.npy', 'SC4462FJ-Hypnogram.npy', 'SC4471FA-Hypnogram.npy', 'SC4472FA-Hypnogram.npy', 'SC4481FV-Hypnogram.npy', 'SC4482FJ-Hypnogram.npy', 'SC4492GJ-Hypnogram.npy', 'SC4501EW-Hypnogram.npy', 'SC4511EJ-Hypnogram.npy', 'SC4512EW-Hypnogram.npy', 'SC4522EM-Hypnogram.npy', 'SC4532EV-Hypnogram.npy', 'SC4541FA-Hypnogram.npy', 'SC4551FC-Hypnogram.npy', 'SC4552FW-Hypnogram.npy', 'SC4561FJ-Hypnogram.npy', 'SC4562FJ-Hypnogram.npy', 'SC4571FV-Hypnogram.npy', 'SC4572FC-Hypnogram.npy', 'SC4581GM-Hypnogram.npy', 'SC4582GP-Hypnogram.npy', 'SC4591GY-Hypnogram.npy', 'SC4592GY-Hypnogram.npy', 'SC4601EC-Hypnogram.npy', 'SC4602EJ-Hypnogram.npy', 'SC4612EA-Hypnogram.npy', 'SC4621EV-Hypnogram.npy', 'SC4622EJ-Hypnogram.npy', 'SC4631EM-Hypnogram.npy', 'SC4641EP-Hypnogram.npy', 'SC4642EP-Hypnogram.npy', 'SC4651EP-Hypnogram.npy', 'SC4652EG-Hypnogram.npy', 'SC4661EJ-Hypnogram.npy', 'SC4662EJ-Hypnogram.npy', 'SC4671GJ-Hypnogram.npy', 'SC4672GV-Hypnogram.npy', 'SC4701EC-Hypnogram.npy', 'SC4702EA-Hypnogram.npy', 'SC4712EA-Hypnogram.npy', 'SC4721EC-Hypnogram.npy', 'SC4722EM-Hypnogram.npy', 'SC4731EM-Hypnogram.npy', 'SC4732EJ-Hypnogram.npy', 'SC4742EC-Hypnogram.npy', 'SC4751EC-Hypnogram.npy', 'SC4752EM-Hypnogram.npy', 'SC4761EP-Hypnogram.npy', 'SC4771GC-Hypnogram.npy', 'SC4772GC-Hypnogram.npy', 'SC4801GC-Hypnogram.npy', 'SC4811GG-Hypnogram.npy', 'SC4812GV-Hypnogram.npy', 'SC4821GC-Hypnogram.npy', 'SC4822GC-Hypnogram.npy']
test_list = ['SC4022EJ-Hypnogram.npy', 'SC4051EC-Hypnogram.npy', 'SC4052EC-Hypnogram.npy', 'SC4111EC-Hypnogram.npy', 'SC4181EC-Hypnogram.npy', 'SC4191EP-Hypnogram.npy', 'SC4222EC-Hypnogram.npy', 'SC4241EC-Hypnogram.npy', 'SC4252EU-Hypnogram.npy', 'SC4261FM-Hypnogram.npy', 'SC4262FC-Hypnogram.npy', 'SC4341FA-Hypnogram.npy', 'SC4342FA-Hypnogram.npy', 'SC4372FC-Hypnogram.npy', 'SC4381FC-Hypnogram.npy', 'SC4431EM-Hypnogram.npy', 'SC4432EM-Hypnogram.npy', 'SC4441EC-Hypnogram.npy', 'SC4442EV-Hypnogram.npy', 'SC4452FW-Hypnogram.npy', 'SC4491GJ-Hypnogram.npy', 'SC4502EM-Hypnogram.npy', 'SC4531EM-Hypnogram.npy', 'SC4542FW-Hypnogram.npy', 'SC4611EG-Hypnogram.npy', 'SC4632EA-Hypnogram.npy', 'SC4711EC-Hypnogram.npy', 'SC4741EA-Hypnogram.npy', 'SC4762EG-Hypnogram.npy', 'SC4802GV-Hypnogram.npy']


def search_annotations_edf(dirname):
    filenames = os.listdir(dirname)
    filenames = [file for file in filenames if file.endswith("Hypnogram.edf")]
    return filenames


def search_signals_edf(dirname):
    filenames = os.listdir(dirname)
    filenames = [file for file in filenames if file.endswith("PSG.edf")]
    return filenames


def search_correct_annotations(dirname, filename):
    search_filename = filename.split('-')[0][:-2]
    file_list = os.listdir(dirname)
    filename = [file for file in file_list if search_filename in file if file.endswith("Hypnogram.edf")]

    return filename


def search_signals_npy(dirname):
    filenames = os.listdir(dirname)
    filenames = [file for file in filenames if file.endswith(".npy")]
    return filenames


def search_correct_annotations_npy(dirname, filename):
    search_filename = filename.split('-')[0][:-2]
    file_list = os.listdir(dirname)
    filename = [file for file in file_list if search_filename in file if file.endswith("npy")]

    return filename


def search_correct_signals_npy(dirname, filename):
    search_filename = filename.split('-')[0][:-2]
    file_list = os.listdir(dirname)
    filename = [file for file in file_list if search_filename in file if file.endswith("npy")]
    return filename

def make_dataset_edf_to_numpy():
    path = 'H:/sleep-edf-database-expanded-1.0.0/sleep-cassette/' # sleep-edf 2013 데이터를 가지고 있는 폴더 명
    annotations_edf_list = search_annotations_edf(path)
    signals_edf_list = search_signals_edf(path)

    print('signals edf file list')
    print(signals_edf_list)

    print('annotations edf file list')
    print(annotations_edf_list)

    print(len(signals_edf_list))
    print(len(annotations_edf_list))

    # index = 0
    # for filename in signals_edf_list:
    #     print('signals file name : %s , annotations file name : %s' % (
    #     filename, search_correct_annotations(path, filename)[0]))
    #     index += 1
    # print(index)

    epoch_size = 30
    sample_rate = 100

    save_path = 'C:/dataset/SleepEDF/'

    save_signals_path = save_path + 'origin_npy/3channel/'
    save_annotations_path = save_path + 'annotations/'

    os.makedirs(save_annotations_path, exist_ok=True)
    os.makedirs(save_signals_path, exist_ok=True)

    for filename in signals_edf_list:
        signals_filename = filename
        annotations_filename = search_correct_annotations(path, filename)[0]

        signals_filename = path + signals_filename
        annotations_filename = path + annotations_filename

        _, _, annotations_header = highlevel.read_edf(annotations_filename)

        label = []
        for ann in annotations_header['annotations']:
            start = ann[0]

            length = ann[1]
            length = int(
                str(length)[2:-1]) // epoch_size  # label은 30초 간격으로 사용할 것이기 때문에 30으로 나눈 값이 해당 sleep stage가 반복된 횟수이다.

            if ann[2] == 'Sleep stage W':
                for time in range(length):
                    label.append(0)
            elif ann[2] == 'Sleep stage 1':
                for time in range(length):
                    label.append(1)
            elif ann[2] == 'Sleep stage 2':
                for time in range(length):
                    label.append(2)
            elif ann[2] == 'Sleep stage 3':
                for time in range(length):
                    label.append(3)
            elif ann[2] == 'Sleep stage 4':
                for time in range(length):
                    label.append(3)
            elif ann[2] == 'Sleep stage R':
                for time in range(length):
                    label.append(4)
            else:
                for time in range(length):
                    label.append(5)
        label = np.array(label)
        signals, _, signals_header = highlevel.read_edf(signals_filename)

        signals_len = len(signals[0]) // sample_rate // epoch_size
        annotations_len = len(label)
        if signals_header['startdate'] == annotations_header['startdate']:
            print("%s file's signal & annotations start time is same" % signals_filename.split('/')[-1])

            if signals_len > annotations_len:
                signals = signals[:3][:annotations_len]
            elif signals_len < annotations_len:
                signals = signals[:3]
                label = label[:signals_len]
            else:
                signals = signals[:3]
            signals = np.array(signals)

            np.save(save_signals_path + signals_filename.split('/')[-1].split('.')[0], signals)
            np.save(save_annotations_path + annotations_filename.split('/')[-1].split('.')[0], label)

            if (len(signals[0]) // sample_rate // epoch_size != len(label)):
                print('signals len : %d / annotations len : %d' % (
                len(signals[0]) // sample_rate // epoch_size, len(label)))

        else:
            print("%s file''s signal & annotations start time is different" % signals_filename.split('/')[-1])


def make_dataset_3channel():
    epoch_size = 30
    sample_rate = 100

    path = 'C:/dataset/SleepEDF/origin_npy/'
    save_path = path+'3channel/'
    os.makedirs(save_path,exist_ok=True)
    signals_npy_list = search_signals_npy(path)

    print(signals_npy_list)

    for filename in signals_npy_list:
        signals_filename = filename

        signals_filename = path + signals_filename

        signals = np.load(signals_filename)

        signals = signals[:3]
        print(signals.shape)

        np.save(save_path + filename, signals)

def check_dataset_truth():
    sample_rate = 100
    epoch_size = 30
    path = 'C:/dataset/SleepEDF/origin_npy/3channel/'
    annotations_path = 'C:/dataset/SleepEDF/annotations/'
    signals_npy_list = search_signals_npy(path)

    print(len(signals_npy_list))

    for filename in signals_npy_list:
        signals_filename = path + filename
        annotations_filename = annotations_path + search_correct_annotations_npy(annotations_path, filename)[0]
        signals = np.load(signals_filename)
        label = np.load(annotations_filename)
        if len(signals[0]) // sample_rate // epoch_size != len(label):
            print('%s is fault' % filename)

def remove_wake():
    fs = 100  # Sampling rate (512 Hz)
    epoch_size = 30
    # data = np.random.uniform(0, 100, 1024)  # 2 sec of data b/w 0.0-100.0
    path = 'C:/dataset/SleepEDF/annotations/'
    signals_path = 'C:/dataset/SleepEDF/origin_npy/3channel/'

    save_annotations_path = path + 'remove_wake/'
    save_signals_path = signals_path + 'remove_wake/'

    os.makedirs(save_annotations_path, exist_ok=True)
    os.makedirs(save_signals_path, exist_ok=True)
    annotations_npy_list = search_signals_npy(path)

    check_index_size = 10

    total_label = np.zeros([6], dtype=int)

    for filename in annotations_npy_list:
        label = np.load(path + filename)
        signals_filename = search_correct_signals_npy(signals_path, filename)[0]

        signals = np.load(signals_path + signals_filename)
        # print(signals.shape)
        for remove_start_index in range(0, len(label), 1):
            # print(np.bincount(label[remove_start_index:(remove_start_index+check_index_size)],minlength=6)[0])
            if (np.bincount(label[remove_start_index:(remove_start_index + check_index_size)], minlength=6)[
                0] != check_index_size):
                break

        for remove_end_index in range(len(label), -1, -1, ):
            # print(np.bincount(label[remove_end_index-check_index_size:(remove_end_index)],minlength=6)[0])
            if (np.bincount(label[remove_end_index - check_index_size:(remove_end_index)], minlength=6)[
                0] != check_index_size and
                    np.bincount(label[remove_end_index - check_index_size:(remove_end_index)], minlength=6)[5] == 0):
                break

        # print('remove start index : %d / remove end index : %d'%(remove_start_index,remove_end_index))
        label = label[remove_start_index:remove_end_index + 1]
        signals = signals[:, remove_start_index * fs * epoch_size:(remove_end_index + 1) * fs * epoch_size]
        # print(np.bincount(label,minlength=6))
        if len(label) == len(signals[0]) // 30 // fs:
            np.save(save_annotations_path + filename.split('.')[0], label)
            np.save(save_signals_path + signals_filename.split('.')[0], signals)
        else:
            print(label.shape)
            print(signals.shape)
        for i in range(6):
            total_label[i] += np.bincount(label, minlength=6)[i]

    print(total_label)

def make_train_testset():
    signals_path = 'C:/dataset/SleepEDF/origin_npy/3channel/remove_wake/'
    annotations_path = 'C:/dataset/SleepEDF/annotations/'
    os.makedirs(signals_path+'train/',exist_ok=True)
    os.makedirs(signals_path + 'test/', exist_ok=True)

    signals_list = os.listdir(signals_path)
    signals_list = [f for f in signals_list if f.endswith('.npy')]

    print(len(signals_list))
    for filename in signals_list:

        annotations_filename = search_correct_annotations_npy(annotations_path,filename)[0]
        if annotations_filename in test_list:
            shutil.move(signals_path + filename, signals_path + 'test/' + filename)
        else:
            shutil.move(signals_path + filename, signals_path + 'train/' + filename)


    train_lists = os.listdir(signals_path+'train/')
    test_lists = os.listdir(signals_path + 'test/')

    print(len(train_lists),len(train_list))
    print(len(test_lists),len(test_list))

def check_train_testset():
    train_path = 'C:/dataset/SleepEDF/origin_npy/3channel/remove_wake/train/'
    test_path = 'C:/dataset/SleepEDF/origin_npy/3channel/remove_wake/test/'
    annotations_path = 'C:/dataset/SleepEDF/annotations/remove_wake/'

    train_list = search_signals_npy(train_path)
    test_list = search_signals_npy(test_path)

    print(train_list)
    print(test_list)

    train_label = np.zeros([6], dtype=int)
    test_label = np.zeros([6], dtype=int)

    for filename in train_list:
        filename = search_correct_annotations_npy(annotations_path, filename)[0]
        label = np.load(annotations_path + filename)

        for i in range(6):
            train_label[i] += np.bincount(label, minlength=6)[i]

    for filename in test_list:
        filename = search_correct_annotations_npy(annotations_path, filename)[0]
        label = np.load(annotations_path + filename)

        for i in range(6):
            test_label[i] += np.bincount(label, minlength=6)[i]

    train_label = train_label / np.sum(train_label) * 100
    test_label = test_label / np.sum(test_label) * 100
    print(train_label)
    print(test_label)

def check_train_testset():
    train_path = 'C:/dataset/SleepEDF/origin_npy/3channel/remove_wake/train/'
    test_path = 'C:/dataset/SleepEDF/origin_npy/3channel/remove_wake/test/'
    annotations_path = 'C:/dataset/SleepEDF/annotations/remove_wake/'

    train_list = search_signals_npy(train_path)
    test_list = search_signals_npy(test_path)

    print(train_list)
    print(test_list)

    train_label = np.zeros([6], dtype=int)
    test_label = np.zeros([6], dtype=int)

    for filename in train_list:
        signals = np.load(train_path + filename)
        filename = search_correct_annotations_npy(annotations_path, filename)[0]
        label = np.load(annotations_path + filename)

        print(len(signals[0])//30//100)
        print(len(label))
        break


#
# make_dataset_edf_to_numpy()
# make_dataset_3channel()
# check_dataset_truth()
# remove_wake()
# make_train_testset()
# check_train_testset()