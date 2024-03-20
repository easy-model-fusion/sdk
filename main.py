import argparse
from sdk.demo import DemoTextConv, DemoTextToImg


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Choose a Model type :'
                                                 ' TextConv or TextToImg')
    subparser = parser.add_subparsers(dest='option')

    conv = subparser.add_parser('TextConv')
    img = subparser.add_parser('TextToImg')
    args = parser.parse_args()

    match args.option:
        case 'TextConv':
            DemoTextConv()
        case ('TextToImg'):
            DemoTextToImg()
        case _:
            DemoTxtToImg()
