import unittest
from sdk.options import Devices
from sdk.sdk.options.options_text_to_image import OptionsTextToImage


class TestOptionsTextToImage(unittest.TestCase):
    def test_init(self):
        device = Devices.GPU
        prompt = "Test prompt"
        image_width = 1024
        image_height = 1024
        num_inference_steps = 50
        timesteps = [1, 2, 3]
        denoising_end = 0.5
        guidance_scale = 5.0
        negative_prompt = "Negative prompt"
        num_images_per_prompt = 2
        eta = 0.1
        generator = None
        latents = None
        prompt_embeds = None
        negative_prompt_embeds = None
        pooled_prompt_embeds = None
        negative_pooled_prompt_embeds = None
        ip_adapter_image = None
        output_type = "pil"
        return_dict = True
        cross_attention_kwargs = {'key': 'value'}
        guidance_rescale = 0.0
        original_size = (800, 600)
        crops_coords_top_left = (100, 100)
        target_size = (512, 512)
        negative_original_size = (800, 600)
        negative_crops_coords_top_left = (100, 100)
        negative_target_size = (512, 512)
        clip_skip = 10
        callback_on_step_end = None
        callback_on_step_end_tensor_inputs = ["latents"]

        options = OptionsTextToImage(
            device=device,
            prompt=prompt,
            image_width=image_width,
            image_height=image_height,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            denoising_end=denoising_end,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            ip_adapter_image=ip_adapter_image,
            output_type=output_type,
            return_dict=return_dict,
            cross_attention_kwargs=cross_attention_kwargs,
            guidance_rescale=guidance_rescale,
            original_size=original_size,
            crops_coords_top_left=crops_coords_top_left,
            target_size=target_size,
            negative_original_size=negative_original_size,
            negative_crops_coords_top_left=negative_crops_coords_top_left,
            negative_target_size=negative_target_size,
            clip_skip=clip_skip,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs
        )

        self.assertEqual(options.device, device)
        self.assertEqual(options.prompt, prompt)
        self.assertEqual(options.image_width, image_width)
        self.assertEqual(options.image_height, image_height)
        self.assertEqual(options.num_inference_steps, num_inference_steps)
        self.assertEqual(options.timesteps, timesteps)
        self.assertEqual(options.denoising_end, denoising_end)
        self.assertEqual(options.guidance_scale, guidance_scale)
        self.assertEqual(options.negative_prompt, negative_prompt)
        self.assertEqual(options.num_images_per_prompt, num_images_per_prompt)
        self.assertEqual(options.eta, eta)
        self.assertEqual(options.generator, generator)
        self.assertEqual(options.latents, latents)
        self.assertEqual(options.prompt_embeds, prompt_embeds)
        self.assertEqual(options.negative_prompt_embeds, negative_prompt_embeds)
        self.assertEqual(options.pooled_prompt_embeds, pooled_prompt_embeds)
        self.assertEqual(options.negative_pooled_prompt_embeds, negative_pooled_prompt_embeds)
        self.assertEqual(options.ip_adapter_image, ip_adapter_image)
        self.assertEqual(options.output_type, output_type)
        self.assertEqual(options.return_dict, return_dict)
        self.assertEqual(options.cross_attention_kwargs, cross_attention_kwargs)
        self.assertEqual(options.guidance_rescale, guidance_rescale)
        self.assertEqual(options.original_size, original_size)
        self.assertEqual(options.crops_coords_top_left, crops_coords_top_left)
        self.assertEqual(options.target_size, target_size)
        self.assertEqual(options.negative_original_size, negative_original_size)
        self.assertEqual(options.negative_crops_coords_top_left, negative_crops_coords_top_left)
        self.assertEqual(options.negative_target_size, negative_target_size)
        self.assertEqual(options.clip_skip, clip_skip)
        self.assertEqual(options.callback_on_step_end, callback_on_step_end)
        self.assertEqual(options.callback_on_step_end_tensor_inputs, callback_on_step_end_tensor_inputs)


if __name__ == '__main__':
    unittest.main()
