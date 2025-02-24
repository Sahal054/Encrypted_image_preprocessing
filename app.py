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

# Uncomment here to have both the server and client in the same terminal
subprocess.Popen(["uvicorn", "server:app"], cwd=REPO_DIR)
time.sleep(3)


def decrypt_output_with_wrong_key(encrypted_image, filter_name):
    """Decrypt the encrypted output using a different private key.
    """
    # Retrieve the filter's deployment path
    filter_path = FILTERS_PATH / f"{filter_name}/deployment"

    # Instantiate the client interface and generate a new private key
    wrong_client = FHEClient(filter_path, filter_name)
    wrong_client.generate_private_and_evaluation_keys(force=True)

    # Deserialize, decrypt and post-process the encrypted output using the new private key
    output_image = wrong_client.deserialize_decrypt_post_process(encrypted_image)

    # For filters that are expected to output black and white images, generate two other random 
    # channels for better display 
    if filter_name in ["black and white", "ridge detection"]:
        # Green channel
        wrong_client.generate_private_and_evaluation_keys(force=True)
        output_image[:, :, 1] = wrong_client.deserialize_decrypt_post_process(encrypted_image)[:, :, 0]

        # Blue channel
        wrong_client.generate_private_and_evaluation_keys(force=True)
        output_image[:, :, 2] = wrong_client.deserialize_decrypt_post_process(encrypted_image)[:, :, 0]

    return output_image


def shorten_bytes_object(bytes_object, limit=500):
    """Shorten the input bytes object to a given length.

    Encrypted data is too large for displaying it in the browser using Gradio. This function
    provides a shorten representation of it.

    Args:
        bytes_object (bytes): The input to shorten
        limit (int): The length to consider. Default to 500.

    Returns:
        str: Hexadecimal string shorten representation of the input byte object. 

    """
    # Define a shift for better display
    shift = 100
    return bytes_object[shift : limit + shift].hex()


def get_client(user_id, filter_name):
    """Get the client API.

    Args:
        user_id (int): The current user's ID.
        filter_name (str): The filter chosen by the user

    Returns:
        FHEClient: The client API.
    """
    return FHEClient(
        FILTERS_PATH / f"{filter_name}/deployment", 
        filter_name,
        key_dir=KEYS_PATH / f"{filter_name}_{user_id}", 
    )


def get_client_file_path(name, user_id, filter_name):
    """Get the correct temporary file path for the client.

    Args:
        name (str): The desired file name.
        user_id (int): The current user's ID.
        filter_name (str): The filter chosen by the user

    Returns:
        pathlib.Path: The file path.
    """
    return CLIENT_TMP_PATH / f"{name}_{filter_name}_{user_id}"


def clean_temporary_files(n_keys=20):
    """Clean keys and encrypted images.

    A maximum of n_keys keys and associated temporary files are allowed to be stored. Once this 
    limit is reached, the oldest files are deleted.

    Args:
        n_keys (int): The maximum number of keys and associated files to be stored. Default to 20.

    """
    # Get the oldest key files in the key directory
    key_dirs = sorted(KEYS_PATH.iterdir(), key=os.path.getmtime)

    # If more than n_keys keys are found, remove the oldest
    user_ids = []
    if len(key_dirs) > n_keys:
        n_keys_to_delete = len(key_dirs) - n_keys
        for key_dir in key_dirs[:n_keys_to_delete]:
            user_ids.append(key_dir.name)
            shutil.rmtree(key_dir)

    # Get all the encrypted objects in the temporary folder
    client_files = CLIENT_TMP_PATH.iterdir()
    server_files = SERVER_TMP_PATH.iterdir()

    # Delete all files related to the ids whose keys were deleted
    for file in chain(client_files, server_files):
        for user_id in user_ids:
            if user_id in file.name:
                file.unlink()


def keygen(filter_name):
    """Generate the private key associated to a filter.

    Args:
        filter_name (str): The current filter to consider.

    Returns:
        (user_id, True) (Tuple[int, bool]): The current user's ID and a boolean used for visual display.

    """
    # Clean temporary files
    clean_temporary_files()

    # Create an ID for the current user
    user_id = numpy.random.randint(0, 2**32)

    # Retrieve the client API
    client = get_client(user_id, filter_name)

    # Generate a private key
    client.generate_private_and_evaluation_keys(force=True)

    # Retrieve the serialized evaluation key. In this case, as circuits are fully leveled, this
    # evaluation key is empty. However, for software reasons, it is still needed for proper FHE
    # execution
    evaluation_key = client.get_serialized_evaluation_keys()

    # Save evaluation_key as bytes in a file as it is too large to pass through regular Gradio
    # buttons (see https://github.com/gradio-app/gradio/issues/1877)
    evaluation_key_path = get_client_file_path("evaluation_key", user_id, filter_name)

    with evaluation_key_path.open("wb") as evaluation_key_file:
        evaluation_key_file.write(evaluation_key)

    return (user_id, True)


def encrypt(user_id, input_image, filter_name):
    """Encrypt the given image for a specific user and filter.

    Args:
        user_id (int): The current user's ID.
        input_image (numpy.ndarray): The image to encrypt.
        filter_name (str): The current filter to consider.

    Returns:
        (input_image, encrypted_image_short) (Tuple[bytes]): The encrypted image and one of its
        representation.

    """
    if user_id == "":
        raise gr.Error("Please generate the private key first.")

    if input_image is None:
        raise gr.Error("Please choose an image first.")
    
    if input_image.shape[-1] != 3:
        raise ValueError(f"Input image must have 3 channels (RGB). Current shape: {input_image.shape}")
    
    # Resize the image if it hasn't the shape (100, 100, 3)
    if input_image.shape != (100 , 100, 3):
        input_image_pil = Image.fromarray(input_image)
        input_image_pil = input_image_pil.resize((100, 100))
        input_image = numpy.array(input_image_pil)

    # Retrieve the client API
    client = get_client(user_id, filter_name)

    # Pre-process, encrypt and serialize the image
    encrypted_image = client.encrypt_serialize(input_image)

    # Save encrypted_image to bytes in a file, since too large to pass through regular Gradio
    # buttons, https://github.com/gradio-app/gradio/issues/1877
    encrypted_image_path = get_client_file_path("encrypted_image", user_id, filter_name)

    with encrypted_image_path.open("wb") as encrypted_image_file:
        encrypted_image_file.write(encrypted_image)

    # Create a truncated version of the encrypted image for display
    encrypted_image_short = shorten_bytes_object(encrypted_image)

    return (resize_img(input_image), encrypted_image_short)


def send_input(user_id, filter_name):
    """Send the encrypted input image as well as the evaluation key to the server.

    Args:
        user_id (int): The current user's ID.
        filter_name (str): The current filter to consider.
    """
    # Get the evaluation key path
    evaluation_key_path = get_client_file_path("evaluation_key", user_id, filter_name)

    if user_id == "" or not evaluation_key_path.is_file():
        raise gr.Error("Please generate the private key first.")

    encrypted_input_path = get_client_file_path("encrypted_image", user_id, filter_name)

    if not encrypted_input_path.is_file():
        raise gr.Error("Please generate the private key and then encrypt an image first.")

    # Define the data and files to post
    data = {
        "user_id": user_id,
        "filter": filter_name,
    }

    files = [
        ("files", open(encrypted_input_path, "rb")),
        ("files", open(evaluation_key_path, "rb")),
    ]

    # Send the encrypted input image and evaluation key to the server
    url = SERVER_URL + "send_input"
    with requests.post(
        url=url,
        data=data,
        files=files,
    ) as response:
        return response.ok


def run_fhe(user_id, filter_name):
    """Apply the filter on the encrypted image previously sent using FHE.

    Args:
        user_id (int): The current user's ID.
        filter_name (str): The current filter to consider.
    """
    data = {
        "user_id": user_id,
        "filter": filter_name,
    }

    # Trigger the FHE execution on the encrypted image previously sent
    url = SERVER_URL + "run_fhe"
    with requests.post(
        url=url,
        data=data,
    ) as response:
        if response.ok:
            return response.json()
        else:
            raise gr.Error("Please wait for the input image to be sent to the server.")


def get_output(user_id, filter_name):
    """Retrieve the encrypted output image.

    Args:
        user_id (int): The current user's ID.
        filter_name (str): The current filter to consider.

    Returns:
        encrypted_output_image_short (bytes): A representation of the encrypted result.

    """
    data = {
        "user_id": user_id,
        "filter": filter_name,
    }

    # Retrieve the encrypted output image
    url = SERVER_URL + "get_output"
    with requests.post(
        url=url,
        data=data,
    ) as response:
        if response.ok:
            encrypted_output = response.content

            # Save the encrypted output to bytes in a file as it is too large to pass through regular
            # Gradio buttons (see https://github.com/gradio-app/gradio/issues/1877)
            encrypted_output_path = get_client_file_path("encrypted_output", user_id, filter_name)

            with encrypted_output_path.open("wb") as encrypted_output_file:
                encrypted_output_file.write(encrypted_output)

            # Decrypt the image using a different (wrong) key for display
            output_image_representation = decrypt_output_with_wrong_key(encrypted_output, filter_name)

            return {encrypted_output_representation: gr.update(value=resize_img(output_image_representation))}

        else:
            raise gr.Error("Please wait for the FHE execution to be completed.")


def decrypt_output(user_id, filter_name):
    """Decrypt the result.

    Args:
        user_id (int): The current user's ID.
        filter_name (str): The current filter to consider.

    Returns:
        (output_image, False, False) ((Tuple[numpy.ndarray, bool, bool]): The decrypted output, as
            well as two booleans used for resetting Gradio checkboxes

    """
    if user_id == "":
        raise gr.Error("Please generate the private key first.")

    # Get the encrypted output path
    encrypted_output_path = get_client_file_path("encrypted_output", user_id, filter_name)

    if not encrypted_output_path.is_file():
        raise gr.Error("Please run the FHE execution first.")

    # Load the encrypted output as bytes
    with encrypted_output_path.open("rb") as encrypted_output_file:
        encrypted_output_image = encrypted_output_file.read()

    # Retrieve the client API
    client = get_client(user_id, filter_name)

    # Deserialize, decrypt and post-process the encrypted output
    decrypted_ouput = client.deserialize_decrypt_post_process(encrypted_output_image)

    # print(f"Decrypted output: {decrypted_ouput.shape=}")
    print(f"derypted image:{decrypted_ouput}")
    
    
    
    return {output_image: gr.update(value=resize_img(decrypted_ouput))}


def resize_img(img, width=256, height=256):
    """Resize the image."""
    if img.dtype != numpy.uint8:
        img = img.astype(numpy.uint8)
    img_pil = Image.fromarray(img)
    # Resize the image
    resized_img_pil = img_pil.resize((width, height))
    # Convert back to a NumPy array
    return numpy.array(resized_img_pil)


demo = gr.Blocks(theme=gr.themes.Citrus(),css="footer{display:none !important}"
                 )


print("Starting the demo...")
with demo:
    gr.Markdown(
        """
      
        <h1 align="center">Image Filtering On Encrypted Data Using Fully Homomorphic Encryption</h1>
       
       
        """
    )

    gr.Markdown("## Client side")
    gr.Markdown("### Step 1: Upload an image. ")
    gr.Markdown(
        f"The image will automatically be resized to shape ({INPUT_SHAPE[0]}x{INPUT_SHAPE[1]}). "
        "The image here, however, is displayed in its original resolution. The true image used "
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
        choices=AVAILABLE_FILTERS, value="inverted", label="Choose your filter", interactive=True, allow_custom_value=True
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
        "decrypted using a different private key. This is only used to visually represent an "
        "encrypted image."
    )
    get_output_button = gr.Button("Receive the encrypted output image from the server.")

    with gr.Row():
        encrypted_output_representation = gr.Image(
            label=f"Encrypted output representation ({INPUT_SHAPE[0]}x{INPUT_SHAPE[1]}):",
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

    # Final input vs output display
    with gr.Row():
        original_image = gr.Image(
            input_image.value,
            label=f"Input image ({INPUT_SHAPE[0]}x{INPUT_SHAPE[1]}):",
            interactive=False,
            height=256,
            width=256,   
        )

        output_image = gr.Image(
            label=f"Output image ({INPUT_SHAPE[0]}x{INPUT_SHAPE[1]}):",
            interactive=False,
            height=256,
            width=256,
        )

    # Button to generate the private key
    keygen_button.click(
        keygen,
        inputs=[filter_name],
        outputs=[user_id, keygen_checkbox],
    )

    # Button to encrypt inputs on the client side
    encrypt_button.click(
        encrypt,
        inputs=[user_id, input_image, filter_name],
        outputs=[original_image, encrypted_input],
    )

    # Button to send the encodings to the server using post method
    send_input_button.click(
        send_input, inputs=[user_id, filter_name], outputs=[send_input_checkbox]
    )

    # Button to send the encodings to the server using post method
    execute_fhe_button.click(run_fhe, inputs=[user_id, filter_name], outputs=[fhe_execution_time])

    # Button to send the encodings to the server using post method
    get_output_button.click(
        get_output, 
        inputs=[user_id, filter_name],
        outputs=[encrypted_output_representation]
    )
    

    # Button to decrypt the output on the client side
    decrypt_button.click(
        decrypt_output,
        inputs=[user_id, filter_name],
        outputs=[output_image, keygen_checkbox, send_input_checkbox],
    )
    print(output_image)

   

demo.launch(share=False)
