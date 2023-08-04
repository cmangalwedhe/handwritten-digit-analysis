from torch import torch
import pygame
from tkinter import *
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

SCREEN_WIDTH, SCREEN_HEIGHT = 28, 28
WIN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Capture User Data")
WIN.fill((255, 255, 255))

model = torch.load('my_mnist_model.pt')

def predict_image(img):
    image = img[0].view(1, 784)
    print(image.dtype)

    with torch.no_grad():
        logps = model(image)

    ps = torch.exp(logps)
    probability = list(ps.numpy()[0])
    predicted_label = probability.index(max(probability))

    return f"I predict that this number is a {predicted_label}"


def transform_image():
    print("HELLO ALL!")
    pygame.image.save(WIN, "C:\\College Projects\\Handwritten Digit Recognition\\images\\screenshot.png")
    image = Image.open("C:\\College Projects\\Handwritten Digit Recognition\\images\\screenshot.png")

    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32)
    ])

    image_tensor = transform(image)
    plt.imshow(image_tensor[0].numpy().squeeze(), cmap='gray_r')
    plt.show()

    return image_tensor


def main():
    run = True
    clock = pygame.time.Clock()
    while run:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    print("does this work? i hope so")
                    print(predict_image(transform_image()))

            if pygame.mouse.get_pressed() == (1, 0, 0):
                x_position, y_position = pygame.mouse.get_pos()
                pygame.draw.rect(WIN, (0, 0, 0), (x_position, y_position, 2, 2))

            pygame.event.pump()

        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    main()
