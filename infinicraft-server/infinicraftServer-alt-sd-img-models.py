import json
import os
from base64 import b64encode, b64decode
from io import BytesIO
import struct
import time

import requests
from flask import Flask, request
from PIL import Image

# --- CONFIGURATION ---
serverPrompt = """
You are an API that takes a combination of items in the form of "item + item" and finds a suitable single JSON output that combines the two. ALWAYS preface your descriptions with "(((centered))), (MCwrld), (pixelated),white background,". These items must follow the Minecraft theme and artwork of pixelated and recognizable. Output ONLY in ENGLISH. Describe as a pixelated image that will be shrunk down to scale.

REQUIRED PARAMETERS:

item (String): The output word (in English). Items can be physical things, or concepts such as time or justice. Make original names for the items, nothing that Minecraft currently has. However, you can combine names.

description (String): A visual description of the item in English, formatted like alt text. Do not include vague ideas. INCLUDE THE ITEM NAME IN ENGLISH IN THE DESCRIPTION.

throwable (Boolean): If the item is throwable or not. Throwable items include small objects that make sense to be thrown.

nutritionalValue (Number): A number between 0 and 1 representing how nutritious the item would be to consume. Items with 0 nutrition are not consumable. If the item should not be eaten, please put 0! Very nutritious items have a value of 1, such as a steak.

attack (Number): A number between 0 and 1 representing the damage that can be dealt by the item. This can also be interpreted as "hardness". Feathers have 0, rocks have 0.5. Most items should have a value above 0.

color (String): The main color of the item. Please keep this as one word, all lowercase, such as blue, green, black, grey, or cyan.

EXAMPLE INPUT:
Animal + Water

EXAMPLE OUTPUT:
{
"item": "Fish",
"description": "A large blue fish with black eyes and a big fin.",
"throwable": true,
"place": true,
"nutritionalValue": 0.8,
"attack": 0.2,
"color": "blue"
}

MISC EXAMPLES:
Player Head + Bone = Body
Show + Sponge = Spongebob
Sand + Sand = Desert
"""

ollamaUrl = "http://localhost:11434/api/chat"  # Your Ollama API endpoint
serverPort = 17707

# Stable Diffusion WebUI Configuration
config = {
    "sd": {
        "webui_url": "http://127.0.0.1:7860",  # Your SD WebUI URL
        "txt2img_endpoint": "/sdapi/v1/txt2img",
        "steps": 32,
        "cfg_scale": 4,
        "width": 512,  # Initial image generation size (pixels)
        "height": 512,  # Initial image generation size (pixels)
        "pixelate_size": 16,  # Final pixel art size (pixels)
        "sampler": "DPM adaptive",
        "negative_prompt": "((cropped)), blur, bad quality, border, background",
        "model": "minecraft_v10.ckpt",  # Specify your desired model here
    }
}
# -----------------------

app = Flask(__name__)

# --- UTILITY FUNCTIONS ---

def load_json_data(filepath: str) -> dict:
    """Loads JSON data from a file, handling file creation and empty file cases."""
    if not os.path.exists(filepath):
        with open(filepath, "w") as file:
            json.dump({}, file)
        print(f"{filepath} did not exist and has been created with an empty JSON object.")
        return {}

    try:
        with open(filepath, "r") as file:
            content = file.read()
            return json.loads(content) if content else {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filepath}")
        return {}

def save_json_data(filepath: str, data: dict) -> bool:
    """Saves JSON data to a file, handling potential errors."""
    try:
        with open(filepath, "w") as file:
            json.dump(data, file, indent=4)
        return True
    except Exception as e:
        print(f"Error: Failed to save data to {filepath} - {e}")
        return False

def get_json_value(key: str) -> any:
    """Retrieves a value from items.json by key."""
    data = load_json_data("items.json")
    return data.get(key, 0)

def add_json_entry(key: str, value: any) -> bool:
    """Adds or updates a key-value pair in items.json."""
    data = load_json_data("items.json")
    data[key] = value
    return save_json_data("items.json", data)

def get_icon(name: str) -> str:
    """Retrieves the base64 encoded icon for a given item name."""
    data = load_json_data("items.json")
    for val in data.values():
        if val.get("name") == name:
            return val.get("iconToSend")
    return ""

def update_icon_by_item_name(name: str, icon_base64: str):
    """Updates the icon for an item in items.json."""
    data = load_json_data("items.json")
    for key, val in data.items():
        if val.get("name") == name:
            add_json_entry(key, {
                "name": name,
                "messageToSend": val.get("messageToSend"),
                "iconToSend": icon_base64,
            })
            return

def wrap_text(text: str, width: int) -> str:
    """Wraps text to a specified width for descriptions."""
    if not text or width <= 0:
        return text

    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + len(current_line) <= width:
            current_line.append(word)
            current_length += len(word)
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)

    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)

def pixelate_image(image: Image.Image, pixel_size: int) -> Image.Image:
    """Pixelates a Pillow Image object to the specified size."""
    image = image.resize((pixel_size, pixel_size), Image.NEAREST)
    return image.convert("RGBA")

def encode_image(image: Image.Image) -> str:
    """Encodes the 16x16 image into a base64 string using the original logic."""
    texture: list[int] = []
    for x in range(16):
        for y in range(16):
            r, g, b, a = image.getpixel((y, x))  # Corrected the pixel coordinates
            if a < 10:
                texture.append(-1)
            else:
                rgb = (r << 16) + (g << 8) + b
                texture.append(rgb)
    encoded_data = b64encode(struct.pack(">{}i".format(len(texture)), *texture)).decode("utf-8")
    return encoded_data

# --- API ROUTES ---

@app.route("/gen", methods=["POST"])
def handle_post_request():
    """Handles item generation requests using Ollama."""
    req = request.json
    saved_item = get_json_value(req["recipe"])

    if saved_item != 0:
        return json.loads(saved_item["messageToSend"])

    try:
        res = requests.post(
            ollamaUrl,
            json={
                "model": "llama3",
                "messages": [
                    {"role": "system", "content": serverPrompt},
                    {"role": "user", "content": req["recipe"]},
                ],
                "stream": False,
            },
            timeout=60  # Increased timeout for Ollama requests
        )
        res.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        response_content = res.json()["message"]["content"]
        print(f"Received response from Ollama:\n{response_content}")

        # Basic structure validation for Ollama response
        if not all(k in response_content for k in ("item", "description", "throwable", "nutritionalValue", "attack", "color")):
            raise ValueError("Ollama response is missing required fields")

        cleaned_return = json.loads(
            "{"
            + response_content
            .replace("\n", "")
            .replace('"', "'")
            .replace("'", "\0")
            .replace("\0", '"')
            .split("{")[1]
            .split("}")[0]
            + "}"
        )
        cleaned_return["description"] = wrap_text(cleaned_return["description"], 25)
        out = json.dumps({"message": json.dumps(cleaned_return)})

        add_json_entry(
            req["recipe"],
            {
                "name": cleaned_return["item"],
                "messageToSend": out,
                "iconToSend": "",  # Placeholder for icon
            },
        )
        return json.loads(out)

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama: {e}")
        return json.dumps({"error": "Failed to communicate with Ollama"}), 500
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Error processing Ollama response: {e}")
        return json.dumps({"error": "Invalid response from Ollama"}), 500

@app.route("/img", methods=["GET"])
def generate_image():
    """Generates or retrieves item icons using Stable Diffusion."""
    description = request.args.get("itemDescription")
    name = description.split(" -")[0]
    color = request.args.get("itemColor")

    saved_icon = get_icon(name)
    if saved_icon:
        return {"success": True, "image": saved_icon}

    try:
        # --- Stable Diffusion Request ---
        sd_payload = {
            "prompt": f"{color} - {description}",
            "steps": config["sd"]["steps"],
            "cfg_scale": config["sd"]["cfg_scale"],
            "width": config["sd"]["width"],
            "height": config["sd"]["height"],
            "sampler_index": config["sd"]["sampler"],
            "send_images": True,
            "save_images": False,
            "negative_prompt": config["sd"]["negative_prompt"],
            "override_settings": {
                "sd_model_checkpoint": config["sd"]["model"]
            },
            "override_settings_restore_afterwards": True
        }

        sd_response = requests.post(
            url=f"{config['sd']['webui_url']}{config['sd']['txt2img_endpoint']}",
            json=sd_payload,
            timeout=60  # Increased timeout for SD image generation
        )
        sd_response.raise_for_status()

        sd_data = sd_response.json()
        if "images" not in sd_data:
            raise ValueError("No images found in Stable Diffusion response")

        # --- Image Processing ---
        base64_image = sd_data["images"][0]
        image_data = b64decode(base64_image)
        image = Image.open(BytesIO(image_data))

        # Save the original image (for debugging)
        os.makedirs("output_images", exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        original_image_path = os.path.join("output_images", f"{name}_{timestamp}_original.png")
        image.save(original_image_path)
        print(f"Original image saved to: {original_image_path}")

        pixelated_image = pixelate_image(image, config["sd"]["pixelate_size"])

        # Save the pixelated image (for debugging)
        pixelated_image_path = os.path.join("output_images", f"{name}_{timestamp}_pixelated.png")
        pixelated_image.save(pixelated_image_path)
        print(f"Pixelated image saved to: {pixelated_image_path}")

        # Encode using the original method
        icon_base64 = encode_image(pixelated_image)

        update_icon_by_item_name(name, icon_base64)
        return {"success": True, "image": icon_base64}

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Stable Diffusion: {e}")
        return json.dumps({"error": "Failed to communicate with Stable Diffusion"}), 500
    except (ValueError, IOError) as e:
        print(f"Error processing image from Stable Diffusion: {e}")
        return json.dumps({"error": "Invalid image data from Stable Diffusion"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=serverPort)