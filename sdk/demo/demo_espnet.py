import scipy
from sdk.options import Devices
from transformers import FastSpeech2ConformerTokenizer, FastSpeech2ConformerModel

from sdk.models import ModelTransformers


class DemoEspnet:

    def __init__(self):
        model_path = "espnet/fastspeech2_conformer"
        tokenizer_path = "espnet/fastspeech2_conformer"

        model_transformers = ModelTransformers(
            model_name="model",
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            task="text-to-audio",
            model_class=FastSpeech2ConformerModel,
            tokenizer_class=FastSpeech2ConformerTokenizer,
            device=Devices.GPU
        )

        model_transformers.load_model()

        result = model_transformers.generate_prompt(
            prompt="Hello, my dog is cute."
        )

        scipy.io.wavfile.write("aud.wav", rate=16000, data=result['audio'][0])
