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

st.set_page_config(page_title="Patient Details", layout="wide")

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
                                model: str = "gpt-4o",
                                max_retries: int = 3,
                                temperature: float = 0.7,
                                max_tokens: int = 4000,
                                output_json_schema=None,
                                image_data: str = None):
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

    # Initialize session state for condition stats
    if "condition_stats" not in st.session_state:
        st.session_state["condition_stats"] = {}

    accuracy_placeholder = st.empty()

    def render_accuracy_info(current_patient_id: str):
        with accuracy_placeholder.container():
            st.markdown(f"### Currently processing: Patient ID **{current_patient_id}**")
            stats_table = []
            total_cases = 0
            total_accurate = 0

            for cond, val in st.session_state["condition_stats"].items():
                stats_table.append({
                    "Condition": cond,
                    "Total Cases": val["total"],
                    "Accurate Predictions": val["accurate"]
                })
                total_cases += val["total"]
                total_accurate += val["accurate"]

            if stats_table:
                df_stats = pd.DataFrame(stats_table)
                st.table(df_stats)

                overall_accuracy = total_accurate / total_cases if total_cases > 0 else 0
                st.markdown(f"### Overall Accuracy: {overall_accuracy:.2%}")
                st.markdown(f"**Total Cases:** {total_cases}")
                st.markdown(f"**Total Correct Predictions:** {total_accurate}")
            else:
                st.info("No cases processed yet.")

    openai_service = OpenAIService()

    try:
        df = pd.read_csv("patients.csv")
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        st.stop()

    for _, row in df.iterrows():
        try:
            s3_bucket, s3_key = parse_s3_url(row["image_url"])
            image_data = get_s3_image(s3_bucket, s3_key)
        except Exception as e:
            st.warning(f"Could not load image from {row['image_url']}: {e}")
            image_data = None

        cv_report, vlm_report, otiz_report = asyncio.run(
            process_image(openai_service, image_data, row["image_url"], row["medical_case_history"])
        )
        
        ground_truth = row['Ground Truth']

        # Update stats for the condition
        stats = st.session_state["condition_stats"].setdefault(
            ground_truth, {"total": 0, "accurate": 0}
        )
        stats["total"] += 1

        # Determine match indicator for this patient
        match_indicator = None
        if otiz_report and "Error" not in otiz_report:
            try:
                otiz_data = json.loads(otiz_report)
                if otiz_data.get('diagnosis'):
                    predicted_condition = otiz_data['diagnosis']
                else:
                    predicted_condition = "Unknown"

                accuracy_json = asyncio.run(
                    openai_service.get_accuracy_report(predicted_condition, ground_truth)
                )
                acc_data = json.loads(accuracy_json)
                if acc_data.get("match") is True:
                    stats["accurate"] += 1
                    match_indicator = True
                else:
                    match_indicator = False
            except Exception as e:
                st.error(f"Error parsing OTIZ output or matching for patient {row['Patient ID']}: {e}")

        render_accuracy_info(row["Patient ID"])

        # Display patient details and reports
        with st.container():
            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                if image_data:
                    st.image(image_data, use_container_width=True)
                else:
                    st.warning("No image to display.")
                st.markdown(f"**Patient ID:** {row['Patient ID']}")
                # Display indicator for match status
                if match_indicator is not None:
                    if match_indicator:
                        st.markdown("<span style='color: green; font-size:24px;'>&#x2705;</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color: red; font-size:24px;'>&#x274C;</span>", unsafe_allow_html=True)
                else:
                    st.info("No prediction match data available.")

                # Insert new block for medical_case_history
                if row.get('medical_case_history'):
                    history_json = json.loads(row['medical_case_history'])
                    with st.expander("Patient Demographics"):
                        demographics = history_json.get('patient_demographics', {})
                        st.write(f"**Age:** {demographics.get('age', 'N/A')}")
                        st.write(f"**Gender:** {demographics.get('gender', 'N/A')}")
                        st.write(f"**Risk Factors:** {', '.join(demographics.get('risk_factors', []))}")
                    with st.expander("Sexual History"):
                        sexual_history = history_json.get('sexual_history', {})
                        st.write(f"**Last Sexual Encounter:** {sexual_history.get('last_sexual_encounter', 'N/A')}")
                        st.write(f"**Partner Count:** {sexual_history.get('partner_count', 'N/A')}")
                        st.write(f"**Partner Genders:** {', '.join(sexual_history.get('partner_genders', []))}")
                        st.write(f"**Sexual Activities:** {', '.join(sexual_history.get('sexual_activities', []))}")
                        st.write(f"**Condom Usage:** {sexual_history.get('condom_usage', 'N/A')}")
                        st.write(f"**STI History:** {sexual_history.get('sti_history', 'N/A')}")
                        st.write(f"**HIV Prevention:** {sexual_history.get('hiv_prevention', 'N/A')}")
                        st.write(f"**Trauma History:** {sexual_history.get('trauma_history', 'N/A')}")
                    with st.expander("Symptoms Timeline"):
                        symptoms = history_json.get('symptoms_timeline', {})
                        st.write(f"**Onset:** {symptoms.get('onset', 'N/A')}")
                        st.write(f"**Progression:** {symptoms.get('progression', 'N/A')}")
                        st.write(f"**Systemic Symptoms:** {symptoms.get('systemic_symptoms', 'N/A')}")
                        st.write(f"**Localized Symptoms:** {symptoms.get('localized_symptoms', 'N/A')}")
                else:
                    st.info("No medical case history available.")

                with st.expander("Image Analysis"):
                    st.write(row['image_analysis'] or "N/A")

                with st.expander("Patient Narrative"):
                    st.write(row['patient_narrative'] or "N/A")
                
            with col2:
                st.markdown("### Original Condition")
                st.markdown(f"**Ground Truth:** {ground_truth}")

                st.markdown("### Diagnosis")
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