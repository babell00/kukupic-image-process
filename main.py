from fastapi import FastAPI, HTTPException, File, UploadFile
from rembg import remove
from fastapi.responses import StreamingResponse
from io import BytesIO
import io

from PIL import Image, ImageFilter, ImageEnhance, ImageChops
import numpy as np
import base64
from fastapi.responses import JSONResponse

app = FastAPI()


@app.post("/remove-background-64/")
async def remove_background_64(file: UploadFile = File(...)):
    # Open the uploaded image file using Pillow
    input_image = Image.open(file.file)

    # Convert the image to bytes for background removal
    img_byte_arr = BytesIO()
    input_image.save(img_byte_arr, format=input_image.format)
    input_image_bytes = img_byte_arr.getvalue()

    # Remove the background using rembg
    output_image_bytes = remove(input_image_bytes)

    # Convert the result back into a PIL image
    output_image = Image.open(BytesIO(output_image_bytes))

    # Convert the image to PNG format and encode it to base64
    output_byte_arr = BytesIO()
    # Save image in PNG format
    output_image.save(output_byte_arr, format="PNG")
    base64_image = base64.b64encode(output_byte_arr.getvalue()).decode('utf-8')

    # Return the base64-encoded image in a JSON response
    return JSONResponse(content={"buffer": base64_image})


@app.post("/remove-background/")
async def remove_background(file: UploadFile = File(...)):
    input_image = Image.open(file.file)
    img_byte_arr = BytesIO()
    input_image.save(img_byte_arr, format=input_image.format)
    input_image_bytes = img_byte_arr.getvalue()

    # Remove background
    output_image_bytes = remove(input_image_bytes)

    # Send back the modified image as a response
    return StreamingResponse(BytesIO(output_image_bytes), media_type="image/png")


@app.post("/blur-background/")
async def blur_background(file: UploadFile = File(...)):
    # Load the uploaded image
    # Ensure image is in RGBA mode for alpha handling
    input_image = Image.open(file.file).convert("RGBA")

    # Determine the format or default to PNG
    image_format = input_image.format if input_image.format else 'PNG'

    # Convert image to bytes
    img_byte_arr = BytesIO()
    # Explicitly specify the format
    input_image.save(img_byte_arr, format=image_format)
    input_image_bytes = img_byte_arr.getvalue()

    # Remove the background
    output_image_bytes = remove(input_image_bytes)

    # Open the image without the background
    output_image = Image.open(io.BytesIO(output_image_bytes)).convert("RGBA")

    # Create a blurred version of the original uploaded image (background)
    blurred_background = input_image.filter(ImageFilter.GaussianBlur(
        radius=15))  # Increase blur for smoother background

    # Create an alpha mask (feather effect for smoothing)
    # Get the alpha channel from the output image
    alpha = output_image.split()[-1]
    # Feather the edges more to smooth the transition
    alpha = alpha.filter(ImageFilter.GaussianBlur(radius=10))

    # Optional: Create a glow effect around the subject for better blending
    # Transparent layer for the glow
    glow_layer = Image.new("RGBA", output_image.size, (0, 0, 0, 0))
    glow_radius = 15  # Adjust the glow size
    glow_alpha = alpha.filter(ImageFilter.GaussianBlur(
        radius=glow_radius))  # Create a glow from the alpha mask

    # Create the glow color, close to the background color (adjust as necessary for your background color)
    # A blueish glow for the background color
    glow_color = Image.new("RGBA", output_image.size, (0, 150, 200, 128))

    # Composite the glow layer with the glow color, using the blurred alpha mask
    glow_layer.paste(glow_color, (0, 0), mask=glow_alpha)

    # Blend the glow with the blurred background
    final_composite = Image.alpha_composite(blurred_background, glow_layer)
    # Paste the subject over the blurred background
    final_composite.paste(output_image, (0, 0), output_image)

    # Save the final image to a BytesIO object to stream it back
    final_image_bytes = BytesIO()
    final_composite.save(final_image_bytes, format="PNG")  # Save in PNG format
    final_image_bytes.seek(0)

    # Return the modified image as a response
    return StreamingResponse(final_image_bytes, media_type="image/png")


def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(*rgb)


def get_dominant_color(image):
    image = image.resize((50, 50))  # Resize to speed up processing
    image = image.convert('RGB')    # Ensure the image is in RGB mode
    pixels = np.array(image)
    pixels = pixels.reshape(-1, 3)
    dominant_color = np.mean(pixels, axis=0)
    dominant_color = [int(c) for c in dominant_color]  # Convert to native int
    return tuple(dominant_color)


def get_complementary_color(rgb):
    return tuple(255 - c for c in rgb)  # Result will be native int


def get_image_metadata(image):
    metadata = {
        'format': image.format,
        'mode': image.mode,
        'size': image.size,  # (width, height)
    }

    # Try to extract EXIF data if present
    try:
        exif_data = image._getexif()
        if exif_data:
            exif = {}
            for tag_id, value in exif_data.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)

                # Handle bytes values
                if isinstance(value, bytes):
                    try:
                        value = value.decode('utf-8', 'ignore')
                    except UnicodeDecodeError:
                        value = value.hex()

                # Ensure the value is JSON serializable
                if isinstance(value, (int, float, str)):
                    exif[tag] = value
                else:
                    exif[tag] = str(value)

            metadata['exif'] = exif
    except AttributeError:
        # _getexif() is not available for this image
        pass
    except Exception as e:
        # Handle other exceptions (e.g., images without EXIF data)
        metadata['exif_error'] = str(e)

    return metadata

# New image-info endpoint with metadata extraction


@app.post("/image-info/")
async def image_info(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        input_image = Image.open(file.file)

        # Process the image to get dominant and complementary colors
        dominant_color = get_dominant_color(input_image)
        comp_color = get_complementary_color(dominant_color)

        # Convert RGB to HEX
        dominant_hex = rgb_to_hex(dominant_color)
        comp_hex = rgb_to_hex(comp_color)

        # Get image metadata
        metadata = get_image_metadata(input_image)

        result = {
            "dominant_color": {
                "rgb": dominant_color,
                "hex": dominant_hex
            },
            "complementary_color": {
                "rgb": comp_color,
                "hex": comp_hex
            },
            "metadata": metadata
        }

        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing image: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
