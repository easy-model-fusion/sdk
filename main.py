import argparse

from src.models.models_management import ModelsManagement
from src.models.model_text_to_image import ModelTextToImage
from src.options.options_text_to_image import OptionsTextToImage, Devices
from demo.demo_main_conv import DemoMainConv
from demo.demo_txt_to_img import DemoMainTxtToImg

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Choose a Model type : TextConv or TxtToImg')
    subparser = parser.add_subparsers(dest='option')

    conv = subparser.add_parser('TextConv')
    img = subparser.add_parser('TxtToImg')
    args = parser.parse_args()

    if args.option == 'TextConv':
        DemoMainConv()
    else:
        DemoMainTxtToImg()
