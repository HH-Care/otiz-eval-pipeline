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
import diagnose_schema
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
    
    async def get_diagnose_report(self, case_history, cv_report, vlm_report):
        """Combine case history, CV report, and VLM report into an OTIZ report."""
        self.logger.info("Generating OTIZ report")
        try:
            diagnose_prompt = f"""
                You are a diagnostic assistant tasked with generating a final diagnosis based on three sources of information: the CV model output, the visual language model (VLM) analysis, and the classical clinical presentation (history and exam). Your role is to implement a robust consensus and conflict‐resolution mechanism that prioritizes clinical history over isolated high-confidence image findings—especially when the CV model misclassifies an image as "Normal" despite the presence of a discharge.

                Context:

                case_history : {str(case_history)}
                cv_report :   {str(cv_report)}
                vlm_report :  {str(vlm_report)}

                Your process should include the following steps:
                1. Evaluate and compare the numeric confidence scores from the CV model with the morphological descriptions provided by the VLM.
                2. Check the clinical history for key features. For example, if the patient presents with a single, nearly painless ulcer with well-defined, firm borders (typical for syphilis), note that clinical context.
                3. If the outputs conflict (e.g., the CV model suggests "Normal" while the VLM and clinical history point towards a specific pathology), trigger an additional verification step. This may include recommending confirmatory lab tests such as RPR/VDRL and treponemal assays, or requesting further diagnostic inputs like histopathological data.
                4. Incorporate a "pain index" or "symptom intensity scale" in the evaluation to differentiate between conditions (e.g., a painless syphilis chancre vs. painful HSV lesions).
                5. Adjust the weight given to each input: elevate clinical history and morphological analysis if they significantly conflict with the high-confidence CV output.
                6. Finally, ensure that any conflicts or uncertainties are resolved by explicitly advising confirmatory testing.

                Your output must be a valid JSON object containing three keys:
                - "final_diagnosis": a concise diagnosis based on the integrated data.
                - "reasoning": a clear explanation of the decision-making process, including how conflicting signals were handled.
                - "recommendation": actionable next steps, such as which confirmatory tests to perform and any adjustments needed for the diagnostic pipeline.

                For example, if the patient has a single painless ulcer with well-defined borders, your JSON output might be:
                
                containing the following keys:
              
                "final_diagnosis": "Syphilis",
                "reasoning": "The CV model misclassified the image as 'Normal', but the VLM detailed analysis and clinical history indicate a classic presentation of a syphilis chancre. The nearly painless ulcer with firm, defined borders strongly suggests syphilis. This discrepancy triggered a secondary evaluation, prioritizing clinical context.",
                "recommendation": "Proceed with confirmatory syphilis screening (RPR/VDRL and treponemal tests) and consider additional testing such as dark-field microscopy or histopathology. Rebalance the weighting of clinical history against high-confidence CV outputs in the diagnostic pipeline."
                

                Make sure your final output is valid JSON.

            """

            # Convert cv_report and vlm_report to strings if they're not already
            cv_report_str = json.dumps(cv_report) if isinstance(cv_report, dict) else str(cv_report)
            vlm_report_str = vlm_report if isinstance(vlm_report, str) else json.dumps(vlm_report)
            
            response = self.client.chat.completions.create(
                model="o3-mini",
                messages=[
                    {
                        "role": "developer",
                        "content": [{"type": "text", "text": diagnose_prompt}]
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                      "name": "diagnostic_evaluation",
                      "strict": True,
                      "schema": {
                        "type": "object",
                        "properties": {
                          "final_diagnosis": {
                            "type": "string",
                            "description": "The final diagnosis made based on the evaluation."
                          },
                          "reasoning": {
                            "type": "string",
                            "description": "The reasoning behind the final diagnosis, detailing the logic and thought process."
                          },
                          "recommendation": {
                            "type": "string",
                            "description": "Recommended next steps based on the diagnosis."
                          }
                        },
                        "required": [
                          "final_diagnosis",
                          "reasoning",
                          "recommendation"
                        ],
                        "additionalProperties": False
                      }
                    }
                  },
                reasoning_effort="high"
            )

            print(response)
            
            content = response.choices[0].message.content

            print(content)
            if content:
                return content
            else:
                return "Error: Unable to generate Diagnosis report"
        except Exception as e:
            self.logger.error(f"Error generating Diagnosis report: {str(e)}")
            return f"Error: {str(e)}"

    async def get_otiz_report(self, diagnose_report):
        """Combine case history, CV report, and VLM report into an OTIZ report."""
        self.logger.info("Generating OTIZ report")
        try:
            analyse_prompt = f"""
            Below is the diagnosis report.Analyse the diagnosis report and provide a diagnosis for the patient.

            Diagnosis Report:
            {diagnose_report}

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

            Consider that diagnoses can be expressed at different levels of specificity. A specific condition (e.g., "Heart Related issue") may be appropriately categorized under a broader term (e.g., "Medical emergency"). 

            If the predicted diagnosis is a broader category that sufficiently covers the ground truth condition, return {{ "match": true }}.
            If it does not cover the condition, return {{ "match": false }}.

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
        image_base64 = base64.b64encode(image_data).decode('utf-8') if image_data else ""
        vlm_report = await openai_service.get_vlm_report(image_base64) if image_data else "No image data for VLM"
        diagnose_report = await openai_service.get_diagnose_report(case_history, cv_report, vlm_report)
        otiz_report = await openai_service.get_otiz_report(diagnose_report)
        
        return cv_report, vlm_report, diagnose_report, otiz_report
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None, None, None

def main():
    st.title("OTIZ Evaluation Dashboard")

    # Initialize session state for condition stats
    if "condition_stats" not in st.session_state:
        st.session_state["condition_stats"] = {}
    # List of all misclassified patients (patient-level info)
    if "misclassified_patients" not in st.session_state:
        st.session_state["misclassified_patients"] = []
    # Track stats by gender
    if "gender_stats" not in st.session_state:
        st.session_state["gender_stats"] = {}

    accuracy_placeholder = st.empty()

    def render_accuracy_info(current_patient_id: str):
        with accuracy_placeholder.container():
            st.markdown(f"### Currently processing: Patient ID **{current_patient_id}**")
            
            # --- Build and display stats by condition ---
            stats_table = []
            total_cases = 0
            total_accurate = 0

            for cond, val in st.session_state["condition_stats"].items():
                # Safely compute accuracy percentage
                if val["total"] > 0:
                    accuracy_percent = (val["accurate"] / val["total"]) * 100
                else:
                    accuracy_percent = 0.0

                stats_table.append({
                    "Condition": cond,
                    "Total Cases": val["total"],
                    "Accurate Predictions": val["accurate"],
                    "Accuracy (%)": round(accuracy_percent, 2)
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

            # --- Display misclassified-patient details if any ---
            if st.session_state["misclassified_patients"]:
                st.markdown("### Misclassified Patients")
                df_misclassified = pd.DataFrame(st.session_state["misclassified_patients"])
                st.table(df_misclassified)

            # --- Build and display stats by gender ---
            if st.session_state["gender_stats"]:
                st.markdown("### Gender Statistics")
                gender_data = []
                for g, vals in st.session_state["gender_stats"].items():
                    gender_data.append({
                        "Gender": g,
                        "Total Cases": vals["total"],
                        "Accurate Predictions": vals["accurate"],
                        "Misclassifications": vals["misclassified"]
                    })
                df_gender = pd.DataFrame(gender_data)
                st.table(df_gender)

    openai_service = OpenAIService()

    # Read your CSV
    try:
        df = pd.read_csv("patients.csv")
    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        st.stop()

    # Main loop over dataframe rows
    for _, row in df.iterrows():
        # Try to load image
        try:
            s3_bucket, s3_key = parse_s3_url(row["image_url"])
            image_data = get_s3_image(s3_bucket, s3_key)
        except Exception as e:
            st.warning(f"Could not load image from {row['image_url']}: {e}")
            image_data = None

        # Retrieve case history JSON for demographics/gender
        demographics = {}
        if row.get('medical_case_history') and not pd.isna(row['medical_case_history']):
            try:
                history_json = json.loads(row['medical_case_history'])
                demographics = history_json.get('patient_demographics', {})
            except (json.JSONDecodeError, TypeError):
                pass
        gender = demographics.get('gender', 'N/A')  # You may adapt how you fetch the gender

        # Process image + predictions
        cv_report, vlm_report, diagnose_report, otiz_report = asyncio.run(
            process_image(openai_service, image_data, row["image_url"], row["medical_case_history"])
        )

        ground_truth = row['Ground Truth']

        # Update stats for the condition
        stats = st.session_state["condition_stats"].setdefault(
            ground_truth, {"total": 0, "accurate": 0}
        )
        stats["total"] += 1

        # Also update stats for the gender
        if gender not in st.session_state["gender_stats"]:
            st.session_state["gender_stats"][gender] = {
                "total": 0,
                "accurate": 0,
                "misclassified": 0
            }
        st.session_state["gender_stats"][gender]["total"] += 1

        match_indicator = None
        predicted_condition = None

        if otiz_report and "Error" not in otiz_report:
            try:
                otiz_data = json.loads(otiz_report)
                predicted_condition = otiz_data.get('diagnosis', 'Unknown')

                # Check accuracy
                accuracy_json = asyncio.run(
                    openai_service.get_accuracy_report(predicted_condition, ground_truth)
                )
                acc_data = json.loads(accuracy_json)
                if acc_data.get("match") is True:
                    stats["accurate"] += 1
                    st.session_state["gender_stats"][gender]["accurate"] += 1
                    match_indicator = True
                else:
                    st.session_state["gender_stats"][gender]["misclassified"] += 1
                    match_indicator = False
            except Exception as e:
                st.error(f"Error parsing OTIZ output or matching for patient {row['Patient ID']}: {e}")

        # If mismatch, add to the misclassified-patients list
        if match_indicator is False:
            st.session_state["misclassified_patients"].append({
                "Patient ID": row["Patient ID"],
                "Ground Truth": ground_truth,
                "Predicted Condition": predicted_condition,
                "Gender": gender
            })

        # Show updated stats & misclassified info
        render_accuracy_info(str(row["Patient ID"]))

        # ------------- Display UI for this patient --------------
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

                if row.get('medical_case_history') and not pd.isna(row['medical_case_history']):
                    try:
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
                    except (json.JSONDecodeError, TypeError):
                        st.info("Invalid medical case history format.")
                else:
                    st.info("No medical case history available.")

                with st.expander("Image Analysis"):
                    st.write(row.get('image_analysis', "N/A"))

                with st.expander("Patient Narrative"):
                    st.write(row.get('patient_narrative', "N/A"))
                
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
                with st.expander("Diagnose Report", expanded=True):
                    st.json(diagnose_report if diagnose_report else "No Diagnose Report")

            st.markdown("---")

if __name__ == "__main__":
    main()