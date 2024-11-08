import requests
import os

def remove_background_from_image(input_image_path, output_image_path, api_key):
    """
    Removes the background from an image using the remove.bg API.
    
    Parameters:
    input_image_path (str): The file path of the input image.
    output_image_path (str): The file path where the processed image will be stored.
    api_key (str): Your remove.bg API key.
    """
    # API endpoint and headers
    url = "https://api.remove.bg/v1.0/removebg"
    headers = {
        "X-Api-Key": api_key
    }

    # Prepare the request payload
    files = {
        "image_file": open(input_image_path, "rb")
    }

    # Send the request and get the response
    response = requests.post(url, files=files, headers=headers, timeout=120)

    # Check if the request was successful
    if response.status_code == requests.codes.ok:
        with open(output_image_path, "wb") as out_file:
            out_file.write(response.content)
        print(f"Image processed and saved to: {output_image_path}")
    else:
        print(f"Error removing background: {response.status_code} - {response.text}")