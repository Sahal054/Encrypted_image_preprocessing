"""A local gradio app that filters images using FHE."""
from PIL import Image
import os
import shutil
import subprocess
import time
import gradio as gr
import numpy
import requests
from itertools import chain

from common import (
    AVAILABLE_FILTERS,
    CLIENT_TMP_PATH,
    SERVER_TMP_PATH,
    EXAMPLES,
    FILTERS_PATH,
    INPUT_SHAPE,
    KEYS_PATH,
    REPO_DIR,
    SERVER_URL,
)
from client_server_interface import FHEClient


def start_server():
    """Start the local FastAPI server used by the Gradio client."""
    return subprocess.Popen(["uvicorn", "server:app"], cwd=REPO_DIR)


def wait_for_server(url, timeout=30, interval=0.5):
    """Wait until the backend is ready to accept requests."""
    deadline = time.time() + timeout
    last_error = None

    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=1)
            if response.ok:
                return
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(interval)

    raise RuntimeError(f"Server did not start at {url} within {timeout} seconds.") from last_error


def decrypt_output_with_wrong_key(encrypted_image, filter_name):
    """Decrypt the encrypted output using a different private key.

    Produces a random-looking image that visually represents the fact that
    only the holder of the correct private key can recover the true result.
    """
    filter_path = FILTERS_PATH / f"{filter_name}/deployment"

    wrong_client = FHEClient(filter_path, filter_name)
    wrong_client.generate_private_and_evaluation_keys(force=True)

    output_image = wrong_client.deserialize_decrypt_post_process(encrypted_image)

    # For filters that output effectively single-channel images, generate
    # random channels so the "garbage" display is more visually distinctive
    if filter_name in ["black and white", "ridge detection", "fingerprint enhance",
                       "sobel horizontal", "sobel vertical", "laplacian"]:
        wrong_client.generate_private_and_evaluation_keys(force=True)
        output_image[:, :, 1] = wrong_client.deserialize_decrypt_post_process(
            encrypted_image
        )[:, :, 0]

        wrong_client.generate_private_and_evaluation_keys(force=True)
        output_image[:, :, 2] = wrong_client.deserialize_decrypt_post_process(
            encrypted_image
        )[:, :, 0]

    return output_image


def shorten_bytes_object(bytes_object, limit=500):
    """Shorten the input bytes object to a given length."""
    shift = 100
    return bytes_object[shift : limit + shift].hex()


def get_client(user_id, filter_name):
    """Get the client API."""
    return FHEClient(
        FILTERS_PATH / f"{filter_name}/deployment",
        filter_name,
        key_dir=KEYS_PATH / f"{filter_name}_{user_id}",
    )


def get_client_file_path(name, user_id, filter_name):
    """Get the correct temporary file path for the client."""
    return CLIENT_TMP_PATH / f"{name}_{filter_name}_{user_id}"


def clean_temporary_files(n_keys=20):
    """Clean keys and encrypted images."""
    key_dirs = sorted(KEYS_PATH.iterdir(), key=os.path.getmtime)

    user_ids = []
    if len(key_dirs) > n_keys:
        n_keys_to_delete = len(key_dirs) - n_keys
        for key_dir in key_dirs[:n_keys_to_delete]:
            user_ids.append(key_dir.name)
            shutil.rmtree(key_dir)

    client_files = CLIENT_TMP_PATH.iterdir()
    server_files = SERVER_TMP_PATH.iterdir()

    for file in chain(client_files, server_files):
        for user_id in user_ids:
            if user_id in file.name:
                file.unlink()


def keygen(filter_name):
    """Generate the private key associated to a filter."""
    clean_temporary_files()

    user_id = numpy.random.randint(0, 2**32)

    client = get_client(user_id, filter_name)
    client.generate_private_and_evaluation_keys(force=True)

    evaluation_key = client.get_serialized_evaluation_keys()

    evaluation_key_path = get_client_file_path("evaluation_key", user_id, filter_name)

    with evaluation_key_path.open("wb") as evaluation_key_file:
        evaluation_key_file.write(evaluation_key)

    return (user_id, True)


def encrypt(user_id, input_image, filter_name):
    """Encrypt the given image for a specific user and filter."""
    if user_id == "":
        raise gr.Error("Please generate the private key first.")

    if input_image is None:
        raise gr.Error("Please choose an image first.")

    # Ensure uint8 for PIL compatibility
    if input_image.dtype != numpy.uint8:
        input_image = input_image.astype(numpy.uint8)

    # Drop alpha channel if present (RGBA → RGB)
    if input_image.ndim == 3 and input_image.shape[-1] == 4:
        input_image = input_image[:, :, :3]

    if input_image.shape[-1] != 3:
        raise ValueError(
            f"Input image must have 3 channels (RGB). Current shape: {input_image.shape}"
        )

    # Resize the image to (100, 100, 3) if needed
    if input_image.shape != (INPUT_SHAPE[0], INPUT_SHAPE[1], 3):
        input_image_pil = Image.fromarray(input_image)
        input_image_pil = input_image_pil.resize(INPUT_SHAPE)
        input_image = numpy.array(input_image_pil)

    client = get_client(user_id, filter_name)

    encrypted_image = client.encrypt_serialize(input_image)

    encrypted_image_path = get_client_file_path("encrypted_image", user_id, filter_name)

    with encrypted_image_path.open("wb") as encrypted_image_file:
        encrypted_image_file.write(encrypted_image)

    encrypted_image_short = shorten_bytes_object(encrypted_image)

    return (resize_img(input_image), encrypted_image_short)


def send_input(user_id, filter_name):
    """Send the encrypted input image and evaluation key to the server."""
    evaluation_key_path = get_client_file_path("evaluation_key", user_id, filter_name)

    if user_id == "" or not evaluation_key_path.is_file():
        raise gr.Error("Please generate the private key first.")

    encrypted_input_path = get_client_file_path("encrypted_image", user_id, filter_name)

    if not encrypted_input_path.is_file():
        raise gr.Error("Please generate the private key and then encrypt an image first.")

    data = {
        "user_id": user_id,
        "filter_name": filter_name,
    }

    files = [
        ("files", open(encrypted_input_path, "rb")),
        ("files", open(evaluation_key_path, "rb")),
    ]

    url = SERVER_URL + "send_input"
    response = requests.post(
        url=url,
        data=data,
        files=files,
    )
    return response.ok


def run_fhe(user_id, filter_name):
    """Apply the filter on the encrypted image previously sent using FHE."""
    data = {
        "user_id": user_id,
        "filter_name": filter_name,
    }

    url = SERVER_URL + "run_fhe"
    response = requests.post(
        url=url,
        data=data,
    )

    if response.ok:
        return response.json()

    raise gr.Error("Please wait for the input image to be sent to the server.")


def get_output(user_id, filter_name):
    """Retrieve the encrypted output image."""
    data = {
        "user_id": user_id,
        "filter_name": filter_name,
    }

    url = SERVER_URL + "get_output"
    response = requests.post(
        url=url,
        data=data,
    )

    if response.ok:
        encrypted_output = response.content

        encrypted_output_path = get_client_file_path("encrypted_output", user_id, filter_name)

        with encrypted_output_path.open("wb") as encrypted_output_file:
            encrypted_output_file.write(encrypted_output)

        output_image_representation = decrypt_output_with_wrong_key(
            encrypted_output, filter_name
        )

        return resize_img(output_image_representation)

    raise gr.Error("Please wait for the FHE execution to be completed.")


def decrypt_output(user_id, filter_name):
    """Decrypt the result."""
    if user_id == "":
        raise gr.Error("Please generate the private key first.")

    encrypted_output_path = get_client_file_path("encrypted_output", user_id, filter_name)

    if not encrypted_output_path.is_file():
        raise gr.Error("Please run the FHE execution first.")

    with encrypted_output_path.open("rb") as encrypted_output_file:
        encrypted_output_image = encrypted_output_file.read()

    client = get_client(user_id, filter_name)

    decrypted_output = client.deserialize_decrypt_post_process(encrypted_output_image)

    return (
        resize_img(decrypted_output),
        gr.update(value=False),
        gr.update(value=False),
    )


def resize_img(img, width=256, height=256):
    """Resize the image for display."""
    if img.dtype != numpy.uint8:
        img = img.astype(numpy.uint8)
    img_pil = Image.fromarray(img)
    resized_img_pil = img_pil.resize((width, height))
    return numpy.array(resized_img_pil)


# ────────────────────────────────────────────────────────────────────── Gradio UI

demo = gr.Blocks(css="footer{display:none !important}")

print("Starting the demo...")
with demo:
    gr.Markdown(
        """
        <h1 align="center">Encrypted Fingerprint Processing Using Fully Homomorphic Encryption</h1>
        """
    )

    gr.Markdown("## Client side")
    gr.Markdown("### Step 1: Upload an image. ")
    gr.Markdown(
        f"The image will automatically be resized to shape ({INPUT_SHAPE[0]}×{INPUT_SHAPE[1]}). "
        "The image here is displayed at its original resolution. The true image used "
        "in this demo can be seen in Step 8."
    )
    with gr.Row():
        input_image = gr.Image(
            value=None, label="Upload an image here.", height=256,
            width=256, sources="upload", interactive=True,
        )
        examples = gr.Examples(
            examples=EXAMPLES, inputs=[input_image], examples_per_page=5, label="Examples to use."
        )

    gr.Markdown("### Step 2: Choose your filter.")
    filter_name = gr.Dropdown(
        choices=AVAILABLE_FILTERS,
        value="ridge detection",
        label="Choose your filter",
        interactive=True,
    )

    gr.Markdown("#### Notes")
    gr.Markdown(
        """
        - The private key is used to encrypt and decrypt the data and will never be shared.
        - No public key is required for these filter operators.
        """
    )

    gr.Markdown("### Step 3: Generate the private key.")
    keygen_button = gr.Button("Generate the private key.")

    with gr.Row():
        keygen_checkbox = gr.Checkbox(label="Private key generated:", interactive=False)

    user_id = gr.Textbox(label="", max_lines=2, interactive=False, visible=False)

    gr.Markdown("### Step 4: Encrypt the image using FHE.")
    encrypt_button = gr.Button("Encrypt the image using FHE.")

    with gr.Row():
        encrypted_input = gr.Textbox(
            label="Encrypted input representation:", max_lines=2, interactive=False
        )

    gr.Markdown("## Server side")
    gr.Markdown(
        "The encrypted value is received by the server. The server can then compute the filter "
        "directly over encrypted values. Once the computation is finished, the server returns "
        "the encrypted results to the client."
    )

    gr.Markdown("### Step 5: Send the encrypted image to the server.")
    send_input_button = gr.Button("Send the encrypted image to the server.")
    send_input_checkbox = gr.Checkbox(label="Encrypted image sent.", interactive=False)

    gr.Markdown("### Step 6: Run FHE execution.")
    execute_fhe_button = gr.Button("Run FHE execution.")
    fhe_execution_time = gr.Textbox(
        label="Total FHE execution time (in seconds):", max_lines=1, interactive=False
    )

    gr.Markdown("### Step 7: Receive the encrypted output image from the server.")
    gr.Markdown(
        "The image displayed here is the encrypted result sent by the server, which has been "
        "decrypted using a **different** private key. This is only used to visually represent an "
        "encrypted image — without the correct key, the output is meaningless noise."
    )
    get_output_button = gr.Button("Receive the encrypted output image from the server.")

    with gr.Row():
        encrypted_output_representation = gr.Image(
            label=f"Encrypted output representation ({INPUT_SHAPE[0]}×{INPUT_SHAPE[1]}):",
            interactive=False,
            height=256,
            width=256,
        )

    gr.Markdown("## Client side")
    gr.Markdown(
        "The encrypted output is sent back to the client, who can finally decrypt it with the "
        "private key. Only the client is aware of the original image and its transformed version."
    )

    gr.Markdown("### Step 8: Decrypt the output.")
    gr.Markdown(
        "The image displayed on the left is the input image used during the demo. The output image "
        "can be seen on the right."
    )
    decrypt_button = gr.Button("Decrypt the output")

    with gr.Row():
        original_image = gr.Image(
            input_image.value,
            label=f"Input image ({INPUT_SHAPE[0]}×{INPUT_SHAPE[1]}):",
            interactive=False,
            height=256,
            width=256,
        )
        output_image = gr.Image(
            label=f"Output image ({INPUT_SHAPE[0]}×{INPUT_SHAPE[1]}):",
            interactive=False,
            height=256,
            width=256,
        )

    # ── Wire up buttons ────────────────────────────────────────────────────
    keygen_button.click(
        keygen,
        inputs=[filter_name],
        outputs=[user_id, keygen_checkbox],
    )

    encrypt_button.click(
        encrypt,
        inputs=[user_id, input_image, filter_name],
        outputs=[original_image, encrypted_input],
    )

    send_input_button.click(
        send_input,
        inputs=[user_id, filter_name],
        outputs=[send_input_checkbox],
    )

    execute_fhe_button.click(
        run_fhe,
        inputs=[user_id, filter_name],
        outputs=[fhe_execution_time],
    )

    get_output_button.click(
        get_output,
        inputs=[user_id, filter_name],
        outputs=[encrypted_output_representation],
    )

    decrypt_button.click(
        decrypt_output,
        inputs=[user_id, filter_name],
        outputs=[output_image, keygen_checkbox, send_input_checkbox],
    )


if __name__ == "__main__":
    start_server()
    wait_for_server(SERVER_URL)
    demo.launch(share=False)