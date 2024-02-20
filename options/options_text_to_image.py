from typing import Optional, Union, List, Dict, Tuple, Any, Callable
from diffusers.image_processor import PipelineImageInput
from options.options import Options, Devices
import torch


class OptionsTextToImage(Options):
    """
    Options for text-to-image models
    """
    prompt: Union[str, List[str]]
    prompt_2: Optional[Union[str, List[str]]] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    num_inference_steps: int = 50
    timesteps: List[int] = None
    denoising_end: Optional[float] = None
    guidance_scale: float = 5.0
    negative_prompt: Optional[Union[str, List[str]]] = None
    negative_prompt_2: Optional[Union[str, List[str]]] = None
    num_images_per_prompt: Optional[int] = 1
    eta: float = 0.0
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None
    latents: Optional[torch.FloatTensor] = None
    prompt_embeds: Optional[torch.FloatTensor] = None
    negative_prompt_embeds: Optional[torch.FloatTensor] = None
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None
    ip_adapter_image: Optional[PipelineImageInput] = None
    output_type: Optional[str] = "pil"
    return_dict: bool = True
    cross_attention_kwargs: Optional[Dict[str, Any]] = None
    guidance_rescale: float = 0.0
    original_size: Optional[Tuple[int, int]] = None
    crops_coords_top_left: Tuple[int, int] = (0, 0)
    target_size: Optional[Tuple[int, int]] = None
    negative_original_size: Optional[Tuple[int, int]] = None
    negative_crops_coords_top_left: Tuple[int, int] = (0, 0)
    negative_target_size: Optional[Tuple[int, int]] = None
    clip_skip: Optional[int] = None
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None
    callback_on_step_end_tensor_inputs: List[str] = ["latents"]

    def __init__(
            self,
            device: Devices,
            prompt: Union[str, List[str]],
            prompt_2: Optional[Union[str, List[str]]] = None,
            image_width: Optional[int] = None,
            image_height: Optional[int] = None,
            num_inference_steps: Optional[int] = None,
            timesteps: List[int] = None,
            denoising_end: Optional[float] = None,
            guidance_scale: Optional[float] = None,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = None,
            eta: Optional[float] = None,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            output_type: Optional[str] = None,
            return_dict: Optional[bool] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: Optional[float] = None,
            original_size: Optional[Tuple[int, int]] = None,
            crops_coords_top_left: Tuple[int, int] = None,
            target_size: Optional[Tuple[int, int]] = None,
            negative_original_size: Optional[Tuple[int, int]] = None,
            negative_crops_coords_top_left: Tuple[int, int] = None,
            negative_target_size: Optional[Tuple[int, int]] = None,
            clip_skip: Optional[int] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = None
    ):
        """
        Initializes the OptionsTextToImage (Comment from stable diffusion xl)
        :param device: The device to use generate prompt
        :param prompt: The prompt to give to the model
        :param image_width: The width of the resulting image (1024 by default)
        :param image_height: The height of the resulting image (1024 by default)
        :param num_inference_steps (`int`, *optional*, defaults to 50):
            The number of denoising steps. More denoising steps usually lead to a higher quality image at the
            expense of slower inference.
        :param timesteps (`List[int]`, *optional*):
            Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
            in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
            passed will be used. Must be in descending order.
        :param denoising_end (`float`, *optional*):
            When specified, determines the fraction (between 0.0 and 1.0) of the total denoising process to be
            completed before it is intentionally prematurely terminated. As a result, the returned sample will
            still retain a substantial amount of noise as determined by the discrete timesteps selected by the
            scheduler. The denoising_end parameter should ideally be utilized when this pipeline forms a part of a
            "Mixture of Denoisers" multi-pipeline setup.
        :param guidance_scale (`float`, *optional*, defaults to 5.0):
            Guidance scale as defined in Classifier-Free Diffusion Guidance. `guidance_scale` is defined as `w`
            of equation 2. of the Imagen Paper. Guidance scale is enabled by setting `guidance_scale > 1`.
            Higher guidance scale encourages generating images closely linked to the text `prompt`, usually at the
            expense of lower image quality.
        :param negative_prompt (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation. If not defined, one has to pass
            `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
            less than `1`).
        :param negative_prompt_2 (`str` or `List[str]`, *optional*):
            The prompt or prompts not to guide the image generation to be sent to `tokenizer_2` and
            `text_encoder_2`. If not defined, `negative_prompt` is used in both text-encoders.
        :param num_images_per_prompt (`int`, *optional*, defaults to 1):
            The number of images to generate per prompt.
        :param eta (`float`, *optional*, defaults to 0.0):
            Corresponds to parameter eta (η) in the DDIM paper. Only applies to [`schedulers.DDIMScheduler`],
            will be ignored for others.
        :param generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
            One or a list of torch generators to make generation deterministic.
        :param latents (`torch.FloatTensor`, *optional*):
            Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
            generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
            tensor will be generated by sampling using the supplied random generator.
        :param prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
            provided, text embeddings will be generated from `prompt` input argument.
        :param negative_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
            argument.
        :param pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
            If not provided, pooled text embeddings will be generated from `prompt` input argument.
        :param negative_pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
            Pre-generated negative pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
            weighting. If not provided, pooled negative_prompt_embeds will be generated from `negative_prompt`
            input argument.
        :param ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
        :param output_type (`str`, *optional*, defaults to `"pil"`):
            The output format of the generated image. Choose between PIL.Image.Image or np.array.
        :param return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a StableDiffusionXLPipelineOutput instead of a plain tuple.
        :param cross_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in diffusers.models.attention_processor.
        :param guidance_rescale (`float`, *optional*, defaults to 0.0):
            Guidance rescale factor proposed by Common Diffusion Noise Schedules and Sample Steps are Flawed.
            `guidance_scale` is defined as `φ` in equation 16. of Common Diffusion Noise Schedules and Sample Steps
            are Flawed. Guidance rescale factor should fix overexposure when using zero terminal SNR.
        :param original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            If `original_size` is not the same as `target_size` the image will appear to be down- or upsampled.
            `original_size` defaults to `(height, width)` if not specified. Part of SDXL's micro-conditioning as
            explained in section 2.2 of the documentation.
        :param crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
            `crops_coords_top_left` can be used to generate an image that appears to be "cropped" from the position
            `crops_coords_top_left` downwards. Favorable, well-centered images are usually achieved by setting
            `crops_coords_top_left` to (0, 0). Part of SDXL's micro-conditioning as explained in section 2.2 of
            the documentation.
        :param target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            For most cases, `target_size` should be set to the desired height and width of the generated image. If
            not specified it will default to `(height, width)`. Part of SDXL's micro-conditioning as explained in
            section 2.2 of the documentation.
        :param negative_original_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            To negatively condition the generation process based on a specific image resolution. Part of SDXL's
            micro-conditioning as explained in section 2.2 of the documentation.
        :param negative_crops_coords_top_left (`Tuple[int]`, *optional*, defaults to (0, 0)):
            To negatively condition the generation process based on specific crop coordinates. Part of SDXL's
            micro-conditioning as explained in section 2.2 of the documentation.
        :param negative_target_size (`Tuple[int]`, *optional*, defaults to (1024, 1024)):
            To negatively condition the generation process based on a target image resolution. It should be as same
            as the `target_size` for most cases. Part of SDXL's micro-conditioning as explained in section 2.2 of
            the documentation.
        :param callback_on_step_end (`Callable`, *optional*):
            A function that calls at the end of each denoising step during the inference. The function is called
            with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
            callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
            `callback_on_step_end_tensor_inputs`.
        :param callback_on_step_end_tensor_inputs (`List`, *optional*):
            The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
            will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
            `._callback_tensor_inputs` attribute of your pipeline class.
        """
        super().__init__(device)
        self.prompt = prompt
        self.prompt_2 = prompt_2
        self.image_width = image_width
        self.image_height = image_height
        if num_inference_steps:
            self.num_inference_steps = num_inference_steps
        self.timesteps = timesteps
        self.denoising_end = denoising_end
        if guidance_scale:
            self.guidance_scale = guidance_scale
        self.negative_prompt = negative_prompt
        self.negative_prompt_2 = negative_prompt_2
        if num_images_per_prompt:
            self.num_images_per_prompt = num_images_per_prompt
        if eta:
            self.eta = eta
        self.generator = generator
        self.latents = latents
        self.prompt_embeds = prompt_embeds
        self.negative_prompt_embeds = negative_prompt_embeds
        self.pooled_prompt_embeds = pooled_prompt_embeds
        self.negative_pooled_prompt_embeds = negative_pooled_prompt_embeds
        self.ip_adapter_image = ip_adapter_image
        if output_type:
            self.output_type = output_type
        if return_dict:
            self.return_dict = return_dict
        self.cross_attention_kwargs = cross_attention_kwargs
        if guidance_rescale:
            self.guidance_rescale = guidance_rescale
        self.original_size = original_size
        if crops_coords_top_left:
            self.crops_coords_top_left = crops_coords_top_left
        self.target_size = target_size
        self.negative_original_size = negative_original_size
        if negative_crops_coords_top_left:
            self.negative_crops_coords_top_left = negative_crops_coords_top_left
        self.negative_target_size = negative_target_size
        self.clip_skip = clip_skip
        self.callback_on_step_end = callback_on_step_end
        if callback_on_step_end_tensor_inputs:
            self.callback_on_step_end_tensor_inputs = callback_on_step_end_tensor_inputs
