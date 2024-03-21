import argparse
from sdk.demo import DemoTextConv, DemoTextToImg, DemoTextGen


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Choose a Model type :'
                                                 ' TextConv, TextToImg '
                                                 'or TextGen')
    subparser = parser.add_subparsers(dest='option')

    conv = subparser.add_parser('TextConv')
    img = subparser.add_parser('TextToImg')
    gen = subparser.add_parser('TextGen')
    args = parser.parse_args()

    match args.option:
        case 'TextConv':
            DemoTextConv()
        case 'TextToImg':
            DemoTextToImg()
        case 'TextGen':
            DemoTextGen()
        case _:
            DemoTextToImg()
