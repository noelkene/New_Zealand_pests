# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import uuid
import copy
import datetime
import subprocess
from decimal import Decimal
from google.cloud import storage
from google.cloud import bigquery
from google.adk.tools import ToolContext
from google import genai
from google.genai import types
import json
import urllib.request
import urllib.parse
import ssl
import certifi


def get_mpi_summary(species_name: str, tool_context: ToolContext) -> dict:
    """
    Searches for the MPI page for a given species and returns a summary.

    Args:
        species_name: The name of the species to search for.
        tool_context: The context for the tool.

    Returns:
        A dictionary containing the summary.
    """
    print(f"--- TOOL: get_mpi_summary called with species_name={species_name} ---")
    client = genai.Client(vertexai=True)

    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"Summarize the MPI page for '{species_name}' in one paragraph. The MPI website is mpi.govt.nz."),
            ],
        )
    ]
    tools = [
        types.Tool(google_search=types.GoogleSearch()),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        seed=0,
        max_output_tokens=8192,
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT", threshold="OFF"
            ),
        ],
        tools=tools,
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    summary = response.text
    print(f"MPI Summary: {summary}")
    tool_context.state['mpi_summary'] = summary
    return {"status": "success", "summary": summary}


def get_weather_forecast(latitude: float, longitude: float, tool_context: ToolContext) -> dict:
    """
    Retrieves the weather forecast for the next 7 days from BigQuery based on latitude and longitude.

    Args:
        latitude: The latitude of the location.
        longitude: The longitude of the location.
        tool_context: The context for the tool.

    Returns:
        A dictionary containing the weather forecast data or an error message.
    """
    print(f"--- TOOL: get_weather_forecast called with lat={latitude}, lon={longitude} ---")
    try:
        client = bigquery.Client()
        query = f"""
            -- Set your target coordinates
            DECLARE target_point GEOGRAPHY DEFAULT ST_GEOGPOINT({longitude}, {latitude});

            SELECT
              f.time AS forecast_timestamp,
              f.2m_temperature - 273.15 AS temperature_celsius, -- Converted from Kelvin
              f.total_precipitation_6hr AS precipitation_6hr_mm,
              -- Calculate Wind Speed in m/s
              SQRT(POW(f.10m_u_component_of_wind, 2) + POW(f.10m_v_component_of_wind, 2)) AS wind_speed_ms,
              -- Calculate meteorological wind direction in degrees
              MOD(CAST(270 - (ATAN2(f.10m_v_component_of_wind, f.10m_u_component_of_wind) * 180 / 3.14159265359) AS NUMERIC), 360) AS wind_direction_degrees,
              ST_DISTANCE(t.geography, target_point) AS distance_from_point_meters
            FROM
              -- Corrected project and table name
              `isv-coe-noelkenehan-00.weathernext_graph_forecasts.59572747_4_0` AS t,
              UNNEST(t.forecast) AS f -- This UNNESTs the forecast array
            WHERE
              -- This geospatial filter makes the query fast by limiting the search area
              ST_DWithin(t.geography, target_point, 20000) -- Find points within 20km
              -- Now, filter the unnested rows by time
              AND f.time BETWEEN CURRENT_TIMESTAMP() AND TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
            ORDER BY
              distance_from_point_meters, -- Get the closest point first
              forecast_timestamp -- Then order its forecasts by time
            LIMIT 5;
        """
        print(f"Executing BigQuery query: {query}")
        query_job = client.query(query)
        results = query_job.result()

        forecasts = [dict(row) for row in results]
        print(f"Found {len(forecasts)} forecast records.")

        # Convert geography object to string for JSON serialization
        for forecast in forecasts:
            if 'geography' in forecast and hasattr(forecast['geography'], 'to_wkt'):
                forecast['geography'] = forecast['geography'].to_wkt()
            # Convert any other non-serializable types
            for key, value in forecast.items():
                if hasattr(value, 'isoformat'): # Handles datetime, timestamp
                    forecast[key] = value.isoformat()
                elif isinstance(value, Decimal):
                    forecast[key] = float(value)


        case_file = tool_context.state.get("caseFile", {})
        case_file["location"] = {"lat": latitude, "lon": longitude}
        tool_context.state["caseFile"] = case_file

        return {"status": "success", "forecasts": forecasts}

    except Exception as e:
        print(f"Error executing BigQuery query: {e}")
        return {"status": "error", "message": f"Failed to retrieve weather forecast: {e}"}


def get_default_insect_image_gcs_uri(tool_context: ToolContext) -> dict:
    """
    Retrieves the GCS URI for the default insect image.
    This tool does not upload an image; it retrieves the URI of a pre-defined image.

    Args:
        tool_context: The context for the tool.

    Returns:
        A dictionary containing the GCS URI of the file.
    """
    print("--- TOOL: get_default_insect_image_gcs_uri called ---")

    # Bucket and object are hardcoded as per user's request to simplify.
    bucket_name = "new-zealand-insects"
    object_name = "insect1.png"
    gcs_uri = f"gs://{bucket_name}/{object_name}"

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name)

        if blob.exists():
            print(f"Found existing image in GCS bucket: {gcs_uri}")
            return {"status": "success", "gcs_uri": gcs_uri}
        else:
            message = f"Default image not found at {gcs_uri}."
            print(message)
            return {"status": "error", "message": message}

    except Exception as e:
        print(f"Error checking for existing image in GCS: {e}")
        return {"status": "error", "message": f"Error accessing GCS: {e}"}


def cross_reference_biosecurity_databases(tool_context: ToolContext) -> dict:
    """
    Cross-references the identified species in the CaseFile against biosecurity databases.
    It reads the species name from the state, simulates a database lookup, and updates the CaseFile.

    Args:
        tool_context: The context for the tool, used to access session state.

    Returns:
        A dictionary confirming the successful analysis and state update.
    """
    print("--- TOOL: cross_reference_biosecurity_databases called ---")
    case_file = tool_context.state.get("caseFile", {})
    print(f"CaseFile received: {case_file}")
    if not case_file or "identification" not in case_file:
        return {"status": "error", "message": "No identification found in CaseFile."}

    species_name = case_file["identification"]["topGuess"]
    print(f"Cross-referencing {species_name} against biosecurity databases.")

    mpi_summary = tool_context.state.get('mpi_summary', '')

    # Mock threat profile
    threat_profile_data = {}
    if "Spodoptera frugiperda" in species_name:
        threat_profile_data = {
            "statusNZ": "Unwanted Organism - Not Established",
            "threatLevel": "HIGH",
            "hosts": ["maize", "sweet corn", "sorghum"],
            "impact": "Fall armyworm can cause significant damage to a wide range of crops, leading to major economic losses.",
            "mpi_summary": mpi_summary
        }
    elif "Brown Marmorated Stink Bug" in species_name or "Halyomorpha halys" in species_name:
        threat_profile_data = {
            "statusNZ": "Unwanted Organism - Not Established",
            "threatLevel": "HIGH",
            "hosts": ["grapes", "apples", "pears", "citrus", "kiwifruit", "corn", "tomatoes", "peppers"],
            "impact": "The Brown Marmorated Stink Bug is a significant threat to New Zealand's horticulture industry. It feeds on a wide range of fruits and vegetables, causing significant economic damage. It is also a social nuisance, as it enters homes in large numbers during autumn and winter.",
            "mpi_summary": mpi_summary
        }
    else:
        threat_profile_data = {"statusNZ": "Benign", "threatLevel": "LOW", "hosts": [], "mpi_summary": mpi_summary}

    # Update the case file in the state
    new_case_file = copy.deepcopy(case_file)
    new_case_file["threatProfile"] = threat_profile_data
    tool_context.state["caseFile"] = new_case_file

    return {"status": "success", "message": f"Threat level for {species_name} assessed as {threat_profile_data['threatLevel']}."}

def generate_and_send_report(tool_context: ToolContext) -> dict:
    """
    Generates a final report from the completed CaseFile, uploads it to GCS, and creates an incident ticket.
    All data is read from the CaseFile in the session state.

    Args:
        tool_context: The context for the tool, used to access session state.

    Returns:
        A dictionary confirming the report was sent and a ticket was created, including the public URL of the report.
    """
    print("--- TOOL: generate_and_send_report called ---")
    case_file = tool_context.state.get("caseFile", {})
    print(f"CaseFile received: {case_file}")
    if not all(k in case_file for k in ["identification", "threatProfile", "riskAssessment", "location"]):
        return {"status": "error", "message": "CaseFile is incomplete for final reporting."}

    # Get image url
    # Using the hardcoded bucket name for consistency with image retrieval
    image_bucket_name = "new-zealand-insects"
    image_url = f"https://storage.mtls.cloud.google.com/{image_bucket_name}/insect1.png"

    # Get coordinates for Google Maps link
    lat = case_file['location']['lat']
    lon = case_file['location']['lon']
    maps_iframe_url = f"https://maps.google.com/maps?q={lat},{lon}&hl=en&z=14&output=embed"

    # Build the report
    report_html = f'''
<!DOCTYPE html>
<html>
<head>
<title>Biosecurity Incident Report</title>
<style>
  body {{ font-family: sans-serif; margin: 2em; }}
  h1 {{ color: #333; }}
  .container {{ max-width: 800px; margin: auto; background: #f9f9f9; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  .section {{ margin-bottom: 20px; }}
  .section h2 {{ border-bottom: 2px solid #eee; padding-bottom: 10px; }}
  .label {{ font-weight: bold; }}
  img {{ max-width: 100%; height: auto; border-radius: 4px; }}
</style>
</head>
<body>
<div class="container">
  <h1>Biosecurity Incident Report: Case {case_file.get('caseId', 'N/A')}</h1>
  
  <div class="section">
    <h2>Risk Assessment</h2>
    <p><i>This assessment is based on a hyper-local and highly accurate weather forecast for the next 7 days, powered by Google's Gencast model.</i></p>
    <p><span class="label">Recommended Alert Level:</span> {case_file['riskAssessment']['alertLevel']}</p>
    <p><span class="label">Summary:</span> {case_file['riskAssessment']['summary']}</p>
    <p><span class="label">Nearby Assets:</span> {', '.join(case_file['riskAssessment']['nearbyAssets'])}</p>
  </div>

  <div class="section">
    <h2>Species Identification</h2>
    <p><span class="label">Top Guess:</span> {case_file['identification']['topGuess']}</p>
    <p><span class="label">Common Name:</span> {case_file['identification']['commonName']}</p>
    <p><span class="label">Confidence:</span> {case_file['identification']['confidence']}</p>
    <img src="{image_url}" alt="Insect Image">
  </div>

  <div class="section">
    <h2>Threat Details</h2>
    <p><span class="label">NZ Status:</span> {case_file['threatProfile']['statusNZ']}</p>
    <p><span class="label">Threat Level:</span> {case_file['threatProfile']['threatLevel']}</p>
    <p><span class="label">Primary Hosts:</span> {', '.join(case_file['threatProfile']['hosts'])}</p>
    <p><span class="label">MPI Summary:</span> {case_file['threatProfile']['mpi_summary']}</p>
  </div>

  <div class="section">
    <h2>Location Data</h2>
    <p><span class="label">Coordinates:</span> {lat}, {lon}</p>
    <iframe
      width="100%"
      height="450"
      style="border:0"
      loading="lazy"
      allowfullscreen
      src="{maps_iframe_url}">
    </iframe>
  </div>

  <div class="section">
    <h2>References</h2>
    <ul>
      <li><a href="https://www.mpi.govt.nz/biosecurity/major-pest-and-disease-threats/brown-marmorated-stink-bug/" target="_blank">MPI: Brown Marmorated Stink Bug</a></li>
      <li><a href="https://www.landcareresearch.co.nz/tools-and-resources/identification/what-is-this-bug/brown-marmorated-stink-bug/" target="_blank">Landcare Research: Brown Marmorated Stink Bug</a></li>
    </ul>
  </div>

</div>
</body>
</html>
'''

    try:
        # Write HTML report to a temporary file
        tmp_report_path = f"/tmp/report-{uuid.uuid4()}.html"
        with open(tmp_report_path, "w") as f:
            f.write(report_html)

        # Upload HTML report to GCS using gcloud
        report_blob_name = f"reports/report-{uuid.uuid4()}.html"
        gcs_uri = f"gs://{image_bucket_name}/{report_blob_name}"

        upload_command = ["gcloud", "storage", "cp", tmp_report_path, gcs_uri]
        subprocess.run(upload_command, check=True)

        # Construct the direct GCS URL for the report
        report_url = f"https://storage.mtls.cloud.google.com/{image_bucket_name}/{report_blob_name}"

        print(f"HTML report uploaded to: {report_url}")

        # Clean up temporary file
        os.remove(tmp_report_path)

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error using gcloud command: {e}")
        return {"status": "error", "message": f"Failed to process report with gcloud: {e}"}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"status": "error", "message": f"An unexpected error occurred: {e}"}


    # Mock sending email and creating ticket
    recipient_list = "noel.kenehan@gmail.com"
    print(f"Emailing report to {recipient_list}...")

    ticket_title = f"Biosecurity Alert: {case_file['identification']['commonName']}"
    print(f"Creating incident ticket: '{ticket_title}'")

    # Update case file status
    new_case_file = copy.deepcopy(case_file)
    new_case_file["status"] = "Reported"
    tool_context.state["caseFile"] = new_case_file

    return {
        "status": "success",
        "message": f"Report generated and available at {report_url}",
        "report_url": report_url
    }

def identify_insect_with_google_search(tool_context: ToolContext) -> dict:
    """Identifies an insect from an image using Google Search.

    Args:
        tool_context: The context for the tool, used to access session state.

    Returns:
        A dictionary containing the identification results.
    """
    case_file = tool_context.state.get("caseFile", {})
    image_uri = case_file.get("imageUri")
    if not image_uri:
        return {"status": "error", "message": "No image URI found in CaseFile."}

    print(f"--- TOOL: identify_insect_with_google_search called with image_uri={image_uri} ---")
    client = genai.Client(vertexai=True)

    model = "gemini-2.5-flash"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(file_uri=image_uri, mime_type="image/jpeg"),
                types.Part.from_text(text="Identify the insect in this image."),
            ],
        )
    ]
    tools = [
        types.Tool(google_search=types.GoogleSearch()),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        seed=0,
        max_output_tokens=8192,
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT", threshold="OFF"
            ),
        ],
        tools=tools,
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    print(f"Identification result: {response.text}")
    return {"status": "success", "identification": response.text}

def update_case_file_with_identification(identification: str, tool_context: ToolContext) -> dict:
    """Updates the CaseFile in the session state with the identification result."""
    print(f"--- TOOL: update_case_file_with_identification called with identification={identification} ---")
    case_file = tool_context.state.get("caseFile", {})

    # Attempt to parse common name from the identification string
    common_name = "Unknown"
    if "is a " in identification:
        try:
            # "The insect in the image is a Spotted Lanternfly (*Lycorma delicatula*)."
            common_name = identification.split("is a ")[1].split(" (")[0].strip()
        except IndexError:
            pass

    case_file["identification"] = {
        "topGuess": identification,
        "commonName": common_name,
        "confidence": "HIGH"  # Mock confidence
    }
    tool_context.state["caseFile"] = case_file
    print(f"CaseFile updated: {case_file}")
    return {"status": "success", "message": "CaseFile updated with identification."}

def update_case_file_with_risk_assessment(risk_assessment: str, tool_context: ToolContext) -> dict:
    """Updates the CaseFile in the session state with the risk assessment result."""
    print(f"--- TOOL: update_case_file_with_risk_assessment called with risk_assessment={risk_assessment} ---")
    case_file = tool_context.state.get("caseFile", {})
    alert_level = "UNKNOWN"
    if "moderate" in risk_assessment.lower():
        alert_level = "MODERATE"
    elif "high" in risk_assessment.lower():
        alert_level = "HIGH"
    elif "critical" in risk_assessment.lower():
        alert_level = "CRITICAL"
    elif "low" in risk_assessment.lower():
        alert_level = "LOW"

    # Extract nearby assets
    nearby_assets = []
    if "vineyards" in risk_assessment.lower():
        nearby_assets.append("vineyards")

    case_file["riskAssessment"] = {
        "summary": risk_assessment,
        "alertLevel": alert_level,
        "nearbyAssets": nearby_assets
    }
    tool_context.state["caseFile"] = case_file
    print(f"CaseFile updated: {case_file}")
    return {"status": "success", "message": "CaseFile updated with risk assessment."}

def create_case_file(image_uri: str, tool_context: ToolContext) -> dict:
    """
    Creates a new CaseFile in the session state with a generated ID and the provided image URI.

    Args:
        image_uri: The GCS URI of the image associated with this case.
        tool_context: The context for the tool.

    Returns:
        A dictionary containing the status and the initial CaseFile details.
    """
    print(f"--- TOOL: create_case_file called with image_uri={image_uri} ---")
    case_id = str(uuid.uuid4())
    initial_case_file = {
        "caseId": case_id,
        "imageUri": image_uri,
        "location": None,
        "identification": None,
        "threatProfile": None,
        "riskAssessment": None,
        "status": "Initiated"
    }
    tool_context.state["caseFile"] = initial_case_file
    print(f"CaseFile created: {initial_case_file}")
    return {"status": "success", "caseFile": initial_case_file}

def update_case_file_with_location(location: str, tool_context: ToolContext) -> dict:
    """
    Updates the CaseFile with geocoded location data (latitude and longitude) using the Google Geocoding API.

    Args:
        location: The human-readable location string (e.g., "1600 Amphitheatre Parkway, Mountain View, CA").
        tool_context: The context for the tool.

    Returns:
        A dictionary containing the status and the updated CaseFile details.
    """
    print(f"--- TOOL: update_case_file_with_location called with location={location} ---")
    case_file = tool_context.state.get("caseFile", {})
    if not case_file:
        return {"status": "error", "message": "No CaseFile found in session state."}

    api_key = os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        return {"status": "error", "message": "GOOGLE_MAPS_API_KEY environment variable not set."}

    params = {
        'address': location,
        'key': api_key
    }
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?{urllib.parse.urlencode(params)}"

    try:
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(geocode_url, context=context) as response:
            geocode_data = json.loads(response.read().decode())

        if geocode_data["status"] == "OK":
            lat = geocode_data["results"][0]["geometry"]["location"]["lat"]
            lng = geocode_data["results"][0]["geometry"]["location"]["lng"]

            case_file["location"] = {
                "description": location,
                "lat": lat,
                "lon": lng
            }
            tool_context.state["caseFile"] = case_file
            print(f"CaseFile updated with location: {case_file['location']}")
            return {"status": "success", "caseFile": case_file}
        else:
            error_message = geocode_data.get("error_message", geocode_data["status"])
            print(f"Geocoding API error: {error_message}")
            return {"status": "error", "message": f"Geocoding failed: {error_message}"}

    except urllib.error.URLError as e:
        print(f"Error calling Geocoding API: {e}")
        return {"status": "error", "message": f"Failed to connect to Geocoding API: {e.reason}"}
    except (KeyError, IndexError) as e:
        print(f"Error parsing Geocoding API response: {e}")
        return {"status": "error", "message": "Failed to parse geocoding response."}
    except json.JSONDecodeError as e:
        print(f"Error decoding Geocoding API response: {e}")
        return {"status": "error", "message": "Invalid response from Geocoding API."}

