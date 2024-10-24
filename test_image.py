from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Function to convert RGB to HEX


def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])


def get_dominant_color(image):
    image = image.resize((50, 50))  # Resize to speed up processing
    image = image.convert('RGB')  # Ensure the image is in RGB mode
    pixels = np.array(image)
    pixels = pixels.reshape(-1, 3)
    dominant_color = np.mean(pixels, axis=0)
    dominant_color = tuple(dominant_color.astype(int))
    return dominant_color


def get_complementary_color(rgb):
    return tuple(255 - val for val in rgb)


def show_complementary_color(image_path):
    image = Image.open(image_path)
    dominant_color = get_dominant_color(image)
    print(f"Dominant Color (RGB): {dominant_color}")

    comp_color = get_complementary_color(dominant_color)
    print(f"Complementary Color (RGB): {comp_color}")

    # Convert RGB to HEX
    dominant_hex = rgb_to_hex(dominant_color)
    comp_hex = rgb_to_hex(comp_color)

    print(f"Dominant Color (HEX): {dominant_hex}")
    print(f"Complementary Color (HEX): {comp_hex}")

    return dominant_color, dominant_hex, comp_color, comp_hex


def display_colors(dominant_color, comp_color, dominant_hex, comp_hex):
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))

    # Display dominant color
    ax[0].imshow(np.full((100, 100, 3), dominant_color, dtype=np.uint8))
    ax[0].set_title(f"Dominant Color\n{dominant_hex}")
    ax[0].axis('off')

    # Display complementary color
    ax[1].imshow(np.full((100, 100, 3), comp_color, dtype=np.uint8))
    ax[1].set_title(f"Complementary Color\n{comp_hex}")
    ax[1].axis('off')

    plt.show()


# Example usage:
image_path = 'transformed_845d669c-58dc-413f-b4ea-b04644e76aa4.jpeg'
dominant_color, dominant_hex, complementary_color, complementary_hex = show_complementary_color(
    image_path)
display_colors(dominant_color, complementary_color,
               dominant_hex, complementary_hex)
