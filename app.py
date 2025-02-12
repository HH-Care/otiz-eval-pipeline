import streamlit as st
import pandas as pd
import boto3
import re
import httpx
import json
import asyncio
import base64
import logging
import os
from typing import Dict
from vlm_schema import vlm_schema
from otiz_schema import otiz_schema
from accuracy_schema import accuracy_schema
from prompts import vlm_agent_prompt, otiz_prompt
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
st.set_page_config(page_title="Patient Details", layout="wide")

# --- Constants ---
llm_agentic_model = "gpt-4o"
llm_max_retries = "3"
llm_temperature = "0.7"
llm_max_tokens = "4000"

def parse_key_value(text):
    """
    Parses a string with keys and values. Pattern:
      Key: Value
    repeated multiple times.
    """
    pattern = r'([A-Za-z ]+):'
    matches = list(re.finditer(pattern, text))
    result = {}
    
    for i, match in enumerate(matches):
        key = match.group(1).strip()
        start = match.end()  
        
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)
        value = text[start:end].strip()
        result[key] = value
    return result

class OpenAIService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = OpenAI(
            default_headers={"OpenAI-Beta": "assistants=v2"}, 
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.prediction_url = "https://otiz-ai-dev.hehealth.net/predict"

    async def get_prediction(self, s3_url: str) -> dict:
        """Get prediction from ML service."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.prediction_url,
                    json={"s3_url": s3_url}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get prediction: {str(e)}")
            raise

    async def get_vlm_report(self, image_data: str) -> str:
        """Generate a VLM report for the given base64-encoded image data."""
        self.logger.info("Generating VLM report for image")
        try:
            vlm_analysis = await self.get_vlm_completion(
                prompt=vlm_agent_prompt,
                output_json_schema=vlm_schema,
                image_data=image_data
            )
            if vlm_analysis:
                return json.dumps(vlm_analysis, indent=2)
            else:
                return "Error: Unable to analyze image"
        except Exception as e:
            self.logger.error(f"Error generating VLM report: {str(e)}")
            return f"Error: {str(e)}"
        
    async def get_vlm_completion(self,
                                prompt: str,
                                model: str = llm_agentic_model,
                                max_retries: int = int(llm_max_retries),
                                temperature: float = float(llm_temperature),
                                max_tokens: int = int(llm_max_tokens),
                                output_json_schema=None,
                                image_data: str = None):
        """
        Calls OpenAI to generate a VLM report based on the provided image and prompt.
        """
        def is_json(content):
            try:
                json.loads(content)
                return True
            except:
                return False

        for attempt in range(max_retries):
            try:
                self.logger.info(f"Attempting VLM analysis (attempt {attempt + 1}/{max_retries})")

                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": prompt}]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ]

                completion_params = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": output_json_schema
                    }
                }

                completion = self.client.chat.completions.create(**completion_params)
                content = completion.choices[0].message.content

                if is_json(content):
                    parsed_content = json.loads(content)
                    return parsed_content
                else:
                    self.logger.warning("VLM response not valid JSON. Retrying...")

            except Exception as e:
                self.logger.error(
                    f"Error in get_vlm_completion (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(2 ** attempt)
        return None

    async def get_otiz_report(self, case_history, cv_report, vlm_report):
        """Combine case history, CV report, and VLM report into an OTIZ report."""
        self.logger.info("Generating OTIZ report")
        try:
            analyse_prompt = f"""
            User's Case History:
            {case_history}

            1) Multi-Class Classification Model (MCL) Analysis:
            {cv_report}

            2) Visual Language Model (VLM) Analysis:
            {vlm_report}
            """

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": otiz_prompt}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": analyse_prompt}]
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": otiz_schema
                },
                temperature=1,
                max_tokens=2048
            )
            
            content = response.choices[0].message.content
            if content:
                return content
            else:
                return "Error: Unable to generate OTIZ report"
        except Exception as e:
            self.logger.error(f"Error generating OTIZ report: {str(e)}")
            return f"Error: {str(e)}"
        
    async def get_accuracy_report(self, otiz_prediction, ground_truth):
        """
        Generate an accuracy report. 
        Returns JSON like: {"match": true} or {"match": false}
        """
        self.logger.info("Generating accuracy report")
        try:
            accuracy_prompt = f"""
            You have provided two diagnoses: a ground truth diagnosis and a predicted diagnosis.

            Check if they describe the same medical condition.

            If they describe the same medical condition,{{ "match": true }}.
            If they don't, return {{ "match": false }}.

            - Ground truth condition: {ground_truth}
            - Predicted condition: {otiz_prediction}
            """

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": accuracy_prompt}]
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": accuracy_schema
                },
                temperature=0.7,
                max_tokens=1024
            )
            
            content = response.choices[0].message.content
            if content:
                return content
            else:
                return '{"match": false}'  # default if no response
        except Exception as e:
            self.logger.error(f"Error generating accuracy report: {str(e)}")
            return '{"match": false}'

def parse_s3_url(s3_url: str):
    """Parse an S3 URL of the form s3://bucket/key. Returns (bucket, key)."""
    pattern = r"s3://([^/]+)/(.+)"
    match = re.match(pattern, s3_url)
    if not match:
        raise ValueError(f"Invalid S3 URL: {s3_url}")
    return match.group(1), match.group(2)

def get_s3_image(bucket: str, key: str) -> bytes:
    """Fetch an image from S3 and return raw bytes."""
    s3 = boto3.client("s3",
                     aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                     aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))
    response = s3.get_object(Bucket=bucket, Key=key)
    return response["Body"].read()

async def process_image(openai_service: OpenAIService, image_data: bytes, s3_url: str, case_history: str):
    """Process image: get CV prediction, VLM report, and OTIZ report."""
    try:
        cv_report = await openai_service.get_prediction(s3_url)
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        vlm_report = await openai_service.get_vlm_report(image_base64)
        otiz_report = await openai_service.get_otiz_report(case_history, cv_report, vlm_report)
        return cv_report, vlm_report, otiz_report
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None, None

def main():
    st.title("OTIZ Evaluation Dashboard")

    # -------------------------
    # 1) Session State Init
    # -------------------------
    if "condition_stats" not in st.session_state:
        # Will hold { condition_string: {"total": X, "accurate": Y}, ... }
        st.session_state["condition_stats"] = {}

    # -------------------------
    # 2) Single Accuracy Table
    #    and "Currently processing"
    # -------------------------
    accuracy_placeholder = st.empty()

    def render_accuracy_info(current_patient_id: str):
        """Render 'Currently processing' + the accuracy stats table."""
        with accuracy_placeholder.container():
            st.markdown(f"### Currently processing: Patient ID **{current_patient_id}**")

            # Build stats table
            stats_table = []
            for cond, val in st.session_state["condition_stats"].items():
                stats_table.append({
                    "Condition": cond,
                    "Total Cases": val["total"],
                    "Accurate Predictions": val["accurate"]
                })

            # Display
            if stats_table:
                df_stats = pd.DataFrame(stats_table)
                st.table(df_stats)
            else:
                st.info("No cases processed yet.")

    # Initialize OpenAI service
    openai_service = OpenAIService()

    # -------------------------
    # 3) Read CSV with patients
    # -------------------------
    try:
        df = pd.read_csv("patients.csv")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv("patients.csv", encoding='latin-1')
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            st.stop()
    except FileNotFoundError:
        st.error("patients.csv file not found. Please ensure it exists.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while reading the CSV file: {str(e)}")
        st.stop()

    # -------------------------
    # 4) Iterate over patients
    # -------------------------
    for _, row in df.iterrows():
        # Attempt to load S3 image
        try:
            s3_bucket, s3_key = parse_s3_url(row["image_url"])
            image_data = get_s3_image(s3_bucket, s3_key)
        except Exception as e:
            st.warning(f"Could not load image from {row['image_url']}: {e}")
            image_data = None

        # Process (CV + VLM + OTIZ)
        cv_report, vlm_report, otiz_report = asyncio.run(
            process_image(openai_service, image_data, row["image_url"], row["History"])
        )
        
        # -------------------------
        # 5) Accuracy Check
        # -------------------------
        ground_truth = row['original condition (simulated)']

        # Ensure we have the dict entry
        stats = st.session_state["condition_stats"].setdefault(
            ground_truth, {"total": 0, "accurate": 0}
        )
        # Increment total
        stats["total"] += 1
        
        if otiz_report and "Error" not in otiz_report:
            try:
                otiz_data = json.loads(otiz_report)
                # Use the diagnosis field directly
                if otiz_data.get('diagnosis'):
                    predicted_condition = otiz_data['diagnosis']
                else:
                    predicted_condition = "Unknown"

                # Accuracy check
                accuracy_json = asyncio.run(
                    openai_service.get_accuracy_report(predicted_condition, ground_truth)
                )
                acc_data = json.loads(accuracy_json)
                if acc_data.get("match") is True:
                    stats["accurate"] += 1
            except Exception as e:
                st.error(f"Error parsing OTIZ output or matching for patient {row['Patient ID']}: {e}")

        # -------------------------------------
        # 6) Re-render "Currently processing" 
        #    + accuracy table at the top
        # -------------------------------------
        render_accuracy_info(row["Patient ID"])

        # -------------------------------------
        # 7) Show patient details below
        # -------------------------------------
        with st.container():
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                if image_data:
                    st.image(image_data, use_container_width=True)
                else:
                    st.warning("No image to display.")
                st.markdown(f"**Patient ID:** {row['Patient ID']}")
                
                with st.expander("Attribution Guide", expanded=True):
                    data = parse_key_value(row['Attribution Guide'])
                    df_attr = pd.DataFrame(list(data.items()), columns=['Key', 'Value'])
                    st.dataframe(df_attr)
                
            with col2:
                st.markdown("### Original Condition")
                st.markdown(f"**Ground Truth:** {ground_truth}")

                st.markdown("###Diagnosis")
                if otiz_report and "Error" not in otiz_report:
                    try:
                        otiz_data = json.loads(otiz_report)
                        if otiz_data.get('diagnosis'):
                            st.markdown(f"**Final Diagnosis:** {otiz_data['diagnosis']}")

                            st.markdown("\n**Differential Diagnoses:**")
                            if otiz_data.get('differential_diagnosis'):
                                for dd in otiz_data['differential_diagnosis']:
                                    st.write(f"- {dd['diagnostic_possibility']}: {dd['confidence_score']:.2%}")
                    except:
                        st.write("No valid differential diagnosis found.")
                
                st.markdown("### Analysis")
                if otiz_report and "Error" not in otiz_report:
                    otiz_data = json.loads(otiz_report)
                    out = otiz_data.get("output", {})
                    st.markdown(f"**Reasoning Summary:** {out.get('reasoning_summary', '')}")
                    st.markdown(f"**Response:** {out.get('response', '')}")

            with col3:
                st.markdown("### Reports")
                with st.expander("OTIZ Report", expanded=True):
                    st.json(otiz_report if otiz_report else "No OTIZ Report")
                with st.expander("VLM Report", expanded=True):
                    st.json(vlm_report if vlm_report else "No VLM Report")
                with st.expander("CV Report", expanded=True):
                    st.json(cv_report if cv_report else "No CV Prediction")

            st.markdown("---")

if __name__ == "__main__":
    main()
