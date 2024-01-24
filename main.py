from sdk import Models

if __name__ == '__main__':
    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    width = 256
    height = 256

    models = Models()
    models.add_model("stabilityai/sdxl-turbo")

    models.load_model("stabilityai/sdxl-turbo")
    image = models.generate_prompt(prompt, width, height)
    image.show()



