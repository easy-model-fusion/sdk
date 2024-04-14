import argparse
from sdk.demo import (DemoTextConv, DemoTextToImg, DemoTextGen,
                      DemoTextToVideo, DemoDreamLikeArt, DemoRunwayml,
                      DemoSalesforce, DemoStabilityaiImg, DemoOpenaiShape,
                      DemoCvssp, DemoLlavaHf, DemoSearchiumAi, DemoT5Base,
                      DemoEspnet, DemoGoogleGemma, DemoDatabricks,
                      DemoPromptHero, DemoRedshift, DemoProtogen)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Choose a Model type :'
                                                 ' TextConv, TextToImg'
                                                 'or TextGen, TextToVideo'
                                                 'or SalesForce, DreamLikeArt'
                                                 'or Runwayml, StabilityaiImg'
                                                 'or OpenaiShape, Cvssp'
                                                 'or LlavaHf, SearchiumAi'
                                                 'or T5base, Espnet, Protogen'
                                                 'or GoogleGemma, Databricks'
                                                 'or PromptHero, Redshift')
    subparser = parser.add_subparsers(dest='option')

    conv = subparser.add_parser('TextConv')
    img = subparser.add_parser('TextToImg')
    gen = subparser.add_parser('TextGen')
    video = subparser.add_parser('TextToVideo')
    Salesforce = subparser.add_parser('SalesForce')
    DreamLikeArt = subparser.add_parser('DreamLikeArt')
    Runwayml = subparser.add_parser('Runwayml')
    StabilityaiImg = subparser.add_parser('StabilityaiImg')
    OpenaiShape = subparser.add_parser('OpenaiShape')
    Cvssp = subparser.add_parser('Cvssp')
    LlavaHf = subparser.add_parser('LlavaHf')
    SearchiumAi = subparser.add_parser('SearchiumAi')
    T5base = subparser.add_parser('T5base')
    Espnet = subparser.add_parser('Espnet')
    Protogen = subparser.add_parser('Protogen')
    GoogleGemma = subparser.add_parser('GoogleGemma')
    Databricks = subparser.add_parser('Databricks')
    PromptHero = subparser.add_parser('PromptHero')
    Redshift = subparser.add_parser('Redshift')

    args = parser.parse_args()

    match args.option:
        case 'TextConv':
            DemoTextConv()
        case 'TextToImg':
            DemoTextToImg()
        case 'DreamLikeArt':
            DemoDreamLikeArt()
        case 'Runwayml':
            DemoRunwayml()
        case 'TextGen':
            DemoTextGen()
        case 'TextToVideo':
            DemoTextToVideo()
        case 'SalesForce':
            DemoSalesforce()
        case 'StabilityaiImg':
            DemoStabilityaiImg()
        case 'OpenaiShape':
            DemoOpenaiShape()
        case 'Cvssp':
            DemoCvssp()
        case 'LlavaHf':
            DemoLlavaHf()
        case 'SearchiumAi':
            DemoSearchiumAi()
        case 'T5base':
            DemoT5Base()
        case 'Espnet':
            DemoEspnet()
        case 'GoogleGemma':
            DemoGoogleGemma()
        case 'Databricks':
            DemoDatabricks()
        case 'PromptHero':
            DemoPromptHero()
        case 'Redshift':
            DemoRedshift()
        case 'Protogen':
            DemoProtogen()
        case _:
            DemoTextToImg()
