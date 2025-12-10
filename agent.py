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
import google.auth
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools.agent_tool import AgentTool
from app import tools

# --- Environment Setup ---
project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
if not project_id:
    try:
        _, project_id = google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError:
        raise ValueError(
            "Could not determine Google Cloud project ID. "
            "Please set the GOOGLE_CLOUD_PROJECT environment variable or "
            "run 'gcloud auth application-default login'."
        )
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

# --- Specialist Agents (Unchanged) ---
identification_agent = LlmAgent(
    name="identification_agent",
    model="gemini-2.5-pro",
    instruction="""Your purpose is to identify the species in the image.
    1. Read the `CaseFile` from the session state to find the `imageUri`.
    2. Call the `identify_insect_with_google_search` tool, passing the `imageUri` as the `image_uri` argument.
    3. The tool will return a JSON object containing an 'identification' key.
    4. You MUST then call the `update_case_file_with_identification` tool, passing the value from the 'identification' key as the `identification` argument.""",
    tools=[tools.identify_insect_with_google_search, tools.update_case_file_with_identification],
)
threat_analyst_agent = LlmAgent(
    name="threat_analyst_agent",
    model="gemini-2.5-flash",
    instruction="Your purpose is to analyze the threat level of the identified species.\n1. Get the species name from the CaseFile.\n2. Call the 'get_mpi_summary' tool with the species name.\n3. Call the 'cross_reference_biosecurity_databases' tool.",
    tools=[tools.get_mpi_summary, tools.cross_reference_biosecurity_databases],
)
risk_assessment_agent = LlmAgent(
    name="risk_assessment_agent",
    model="gemini-2.5-pro",
    instruction="""Your purpose is to assess the real-world risk of insect spread.
    1. Get the latitude and longitude from the `CaseFile`.
    2. Call the `get_weather_forecast` tool with the latitude and longitude.
    3. Analyze the weather forecast results to assess the risk of spread. Consider the wind speed and direction to determine if the insect could be carried to nearby sensitive areas.
    4. Call the `update_case_file_with_risk_assessment` tool with your risk assessment summary.
    """,
    tools=[tools.get_weather_forecast, tools.update_case_file_with_risk_assessment],
)
mpi_reporting_agent = LlmAgent(
    name="mpi_reporting_agent",
    model="gemini-2.5-flash",
    instruction="You are the final step. Call the 'generate_and_send_report' tool to create and distribute the final report. After the tool returns the 'report_url', you MUST output this URL to the user.",
    tools=[tools.generate_and_send_report],
)

# --- Analysis Pipeline Agent ---
analysis_pipeline_agent = SequentialAgent(
    name="analysis_pipeline_agent",
    description="This agent runs the full biosecurity analysis pipeline. It takes an existing CaseFile with an image URI and runs the identification, threat, risk, and reporting agents in sequence.",
    sub_agents=[
        identification_agent,
        threat_analyst_agent,
        risk_assessment_agent,
        mpi_reporting_agent,
    ]
)

# --- Root Orchestrator Agent ---
root_agent = LlmAgent(
    name="biosecurity_orchestrator",
    model="gemini-2.5-flash",
    instruction="""You are the primary orchestrator for the Bio-Secure NZ system. Your job is to handle initial user requests and kick off the analysis pipeline.

    Workflow:
    1.  **Start Analysis:** When the user asks to analyze an insect, your first job is to call the `upload_latest_image_to_gcs` tool. This tool will retrieve the URI for a pre-defined image.
    2.  **Create CaseFile:** After the tool returns a GCS URI, create a `CaseFile` object in the session state with a `caseId`, and the `imageUri`.
    3.  **Location:** Check if the user has provided a location in their message. If so, add it to the `CaseFile`. If not, ask the user for the location.
    4.  **Invoke Pipeline:** Once the `CaseFile` is in the state and has a location, call the `analysis_pipeline_agent` tool to run the full analysis.
    5.  **No Image:** If the user has not provided an image, ask them to upload one.
    """,
    tools=[
        tools.upload_latest_image_to_gcs,
        AgentTool(analysis_pipeline_agent)
    ],
)
