"""
This module provides functions to interact with part of the Open Polar Server (OPS) API
that do not require authentication.
It includes functions to retrieve segment IDs, layer points, and segment metadata.
"""

import base64
import requests
import json
import urllib.parse
import time

ops_base_url = "https://ops.cresis.ku.edu/ops"


def get_layer_points(segment_name : str, season_name : str, location=None, layer_names=None, include_geometry=True, raise_errors=True):
    """
    Get layer points for a segment from the OPS API.

    Parameters
    ----------
    segment_name : str
        The segment name
    season_name : str
        The season name
    location : str, optional
        The location, either 'arctic' or 'antarctic'. If None, inferred from season_name.
    layer_names : list of str, optional
        List of layer names to retrieve. If None, retrieves all layers.
    include_geometry : bool, optional
        Whether to include geometry information in the response. Default is True.

    Returns
    -------
    dict or None
        API response as JSON containing layer points data, or None if request fails.
        
    Raises
    ------
    ValueError
        If neither segment_id nor both segment_name and season_name are provided.
    """

    if location is None:
        if "antarctica" in season_name.lower():
            location = "antarctic"
        elif "greenland" in season_name.lower():
            location = "arctic"
        else:
            raise ValueError("Location could not be inferred from the season name. Please specify 'arctic' or 'antarctic' explicitly.")

    data_payload = {
        "properties": {
            "location": location,
            "season": season_name,
            "segment": segment_name
        }
    }

    if include_geometry:
        data_payload["properties"]["return_geom"] = 'geog'
    
    # Add layer names if specified
    if layer_names:
        data_payload["properties"]["lyr_name"] = layer_names

    return _ops_api_request(f"/get/layer/points", data_payload, request_type='POST')

def get_layer_metadata():
    """
    Get the mapping of layer IDs to groups and names from the OPS API.

    Returns
    -------
    dict or None
        API response as JSON containing layer metadata, or None if request fails.
    """

    return _ops_api_request(f"/get/layers", {}, request_type='POST')

def get_segment_metadata(segment_name : str, season_name : str):
    """
    Get segment metadata from the OPS API.

    Parameters
    ----------
    segment_name : str, optional
        The segment name (alternative to segment_id).
    season_name : str, optional
        The season name (required if using segment_name).

    Returns
    -------
    dict or None
        API response as JSON containing segment metadata, or None if request fails.
        
    Raises
    ------
    ValueError
        If neither segment_id nor both segment_name and season_name are provided.
    """

    data_payload = {
        "properties": {
            "segment": segment_name,
            "season": season_name
        }
    }

    return _ops_api_request(f"/get/segment/metadata", data_payload)


def _ops_api_request(path, data, request_type='POST', headers=None, base_url=ops_base_url, retries=3, job_timeout=200, debug=False, initial_retry_time=1):
    """
    Helper function to make a POST request to the OPS API.

    Parameters
    ----------
    path : str
        The API endpoint path.
    data : dict
        The data payload to send in the request.
    headers : dict, optional
        Additional headers to include in the request.
    base_url : str, optional
        The base URL for the OPS API. Default is ops_base_url.
    retries : int, optional
        Number of retry attempts for failed requests. Default is 3.

    Returns
    -------
    dict or None
        API response as JSON, or None if request fails after retries.
    """
    
    url = f"{base_url}/{path.lstrip('/')}"


    # Encode the JSON data
    encoded_data = urllib.parse.quote(json.dumps(data))
    form_data = f"app=rds&data={encoded_data}"

    # Create basic auth header for anonymous user
    if headers is None:
        credentials = base64.b64encode(b"anonymous:anonymous").decode("ascii")
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'Authorization': f'Basic {credentials}',
            'Cookie': 'userName=anonymous; isAuthenticated=0'
        }
    
    try:
        # Make the request
        if request_type == 'POST':
            if debug:
                print(f"Making POST request to {url} with data: {form_data}")
            response = requests.post(url, data=form_data, headers=headers)
        elif request_type == 'GET':
            if debug:
                print(f"Making GET request to {url}")
            response = requests.get(url, headers=headers)

        # Check if request was successful
        response.raise_for_status()
        
        response_json = response.json()

        if debug:
            print(f"Received response with status: {response_json['status']}")

        if response_json['status'] == 303: # Indicates that the request was processed as a background task
            task_id = response_json['data']['task_id']
            task_start_time = time.time()
            if debug:
                print(f"Task {task_id} started.")
            while time.time() - task_start_time < job_timeout:
                # Check the status of the background task
                status_response = _ops_api_request(f"/get/status/{urllib.parse.quote(task_id)}", {}, request_type='GET', headers=headers, base_url=base_url, retries=retries)
                
                if debug:
                    print(f"Checking status for task {task_id}: {status_response}")

                if status_response and status_response['status'] != 503:
                    return status_response

                time.sleep(initial_retry_time)  # Wait before retrying
                initial_retry_time *= 2  # Exponential backoff

            raise TimeoutError(f"Task {task_id} timed out after {job_timeout} seconds.")
        
        return response_json

    except requests.exceptions.HTTPError as e:
        if debug:
            print(f"HTTP error occurred: {response.status_code}")
        if response.status_code == 504:
            print(f"Gateway timeout occurred. Retrying in {initial_retry_time} seconds...")
            time.sleep(initial_retry_time)
            return _ops_api_request(path, data, request_type=request_type, headers=headers, base_url=base_url, retries=retries-1, initial_retry_time=initial_retry_time*2)
        raise e
