import argparse
from demo import DemoTextConv, DemoTxtToImg


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Choose a Model type :'
                                                 ' TextConv or TxtToImg')
    subparser = parser.add_subparsers(dest='option')

    conv = subparser.add_parser('TextConv')
    img = subparser.add_parser('TxtToImg')
    args = parser.parse_args()

    match args.option:
        case 'TextConv':
            DemoTextConv()
        case ('TxtToImg'):
            DemoTxtToImg()
        case _:
            DemoTxtToImg()
