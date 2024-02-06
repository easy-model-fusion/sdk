from diffusers import DiffusionPipeline, StableDiffusionXLPipeline
import torch
from src.models.models import Models
from src.options.options_text_to_image import OptionsTextToImage, Devices


class ModelsTextToImage(Models):
    """
    This class implements methods to generate images with a text prompt
    """
    pipeline: StableDiffusionXLPipeline
    model_name: str
    loaded: bool

    def __init__(self, model_name: str):
        """
        Initializes the ModelsTextToImage class
        :param model_name: The name of the model
        """
        super().__init__(model_name)
        self.loaded = False
        self.create_pipeline()

    def create_pipeline(self):
        """
        Creates the pipeline to load on the device
        """
        if self.loaded:
            return

        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            use_safetensors=True,
            add_watermarker=False,
            variant="fp16",
        )

    def load_model(self, option: OptionsTextToImage) -> bool:
        """
        Load this model on the given device
        :param option: The options with the device
        :return: True if the model is successfully loaded
        """
        if self.loaded:
            return True
        if option.device == Devices.RESET:
            return False
        self.pipeline.to(option.device.value)
        self.loaded = True
        return True

    def unload_model(self) -> bool:
        """
        Unloads the model
        :return: True if the model is successfully unloaded
        """
        if not self.loaded:
            return False
        self.pipeline.to(device=Devices.RESET)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        self.loaded = False
        return True

    def generate_prompt(self, options: OptionsTextToImage):
        """
        Generates the prompt with the given option
        :param options: The options of text to image model
        :return: An object image resulting from the model
        """
        return self.pipeline(
            prompt=options.prompt,
            prompt_2=options.prompt_2,
            width=options.image_width,
            height=options.image_height,
            num_inference_steps=options.num_inference_steps,
            timesteps=options.timesteps,
            denoising_end=options.denoising_end,
            guidance_scale=options.guidance_scale,
            negative_prompt=options.negative_prompt,
            negative_prompt_2=options.negative_prompt_2,
            num_images_per_prompt=options.num_images_per_prompt,
            eta=options.eta,
            generator=options.generator,
            latents=options.latents,
            prompt_embeds=options.prompt_embeds,
            negative_prompt_embeds=options.negative_prompt_embeds,
            pooled_prompt_embeds=options.pooled_prompt_embeds,
            negative_pooled_prompt_embeds=options.negative_pooled_prompt_embeds,
            ip_adapter_image=options.ip_adapter_image,
            output_type=options.output_type,
            return_dict=options.return_dict,
            cross_attention_kwargs=options.cross_attention_kwargs,
            guidance_rescale=options.guidance_rescale,
            original_size=options.original_size,
            crops_coords_top_left=options.crops_coords_top_left,
            target_size=options.target_size,
            negative_original_size=options.negative_original_size,
            negative_target_size=options.negative_target_size,
            clip_skip=options.clip_skip,
            callback_on_step_end=options.callback_on_step_end,
            callback_on_step_end_tensor_inputs=options.callback_on_step_end_tensor_inputs
        ).images[0]
