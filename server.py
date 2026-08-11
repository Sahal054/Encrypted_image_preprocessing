"""Server that will listen for GET and POST requests from the client."""

import time
from typing import List
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from common import FILTERS_PATH, SERVER_TMP_PATH, AVAILABLE_FILTERS
from client_server_interface import FHEServer

# Load the server objects related to all currently available filters once and for all
FHE_SERVERS = {
    filter_name: FHEServer(FILTERS_PATH / f"{filter_name}/deployment")
    for filter_name in AVAILABLE_FILTERS
}


def get_server_file_path(name, user_id, filter_name):
    """Get the correct temporary file path for the server."""
    return SERVER_TMP_PATH / f"{name}_{filter_name}_{user_id}"


app = FastAPI()


@app.get("/")
def root():
    return {"message": "Welcome to the Encrypted Fingerprint Processing FHE Server!"}


@app.post("/send_input")
def send_input(
    user_id: str = Form(),
    filter_name: str = Form(),
    files: List[UploadFile] = File(),
):
    """Receive the encrypted input image and evaluation key from the client."""
    encrypted_image_path = get_server_file_path("encrypted_image", user_id, filter_name)
    evaluation_key_path = get_server_file_path("evaluation_key", user_id, filter_name)

    with encrypted_image_path.open("wb") as encrypted_image, evaluation_key_path.open(
        "wb"
    ) as evaluation_key:
        encrypted_image.write(files[0].file.read())
        evaluation_key.write(files[1].file.read())


@app.post("/run_fhe")
def run_fhe(
    user_id: str = Form(),
    filter_name: str = Form(),
):
    """Execute the filter on the encrypted input image using FHE."""
    encrypted_image_path = get_server_file_path("encrypted_image", user_id, filter_name)
    evaluation_key_path = get_server_file_path("evaluation_key", user_id, filter_name)

    with encrypted_image_path.open("rb") as encrypted_image_file, evaluation_key_path.open(
        "rb"
    ) as evaluation_key_file:
        encrypted_image = encrypted_image_file.read()
        evaluation_key = evaluation_key_file.read()

    fhe_server = FHE_SERVERS[filter_name]

    start = time.time()
    encrypted_output_image = fhe_server.run(encrypted_image, evaluation_key)
    fhe_execution_time = round(time.time() - start, 2)

    encrypted_output_path = get_server_file_path("encrypted_output", user_id, filter_name)

    with encrypted_output_path.open("wb") as encrypted_output:
        encrypted_output.write(encrypted_output_image)

    return JSONResponse(content=fhe_execution_time)


@app.post("/get_output")
def get_output(
    user_id: str = Form(),
    filter_name: str = Form(),
):
    """Retrieve the encrypted output image."""
    encrypted_output_path = get_server_file_path("encrypted_output", user_id, filter_name)

    with encrypted_output_path.open("rb") as encrypted_output_file:
        encrypted_output = encrypted_output_file.read()

    return Response(encrypted_output)