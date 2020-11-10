from include.header import *
from utils.function.function import *
from utils.function.loss_fn import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='테스트')
    parser.add_argument("-mode", required=False, help='-train & -test',default='Something')

    args = parser.parse_args()

    if args.mode == 'None':
        print('Please Select Mode!')
        exit(1)
    if args.mode == 'train':
        print('Train Mode')
    if args.mode == 'Something':
        test()
        print('Something!!')




