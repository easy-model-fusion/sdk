import argparse
from demo.demo_text_conv import DemoTextConv
from demo.demo_txt_to_img import DemoMainTxtToImg

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
            DemoMainTxtToImg()
        case _:
            DemoMainTxtToImg()
