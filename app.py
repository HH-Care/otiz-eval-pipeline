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
from prompts import skin_filter_agent_prompt, otiz_prompt
from openai import OpenAI
from dotenv import load_dotenv
from skin_filter_schema import otiz_skin_filter_schema
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

    async def get_lvrm_report(self, image_data: str, clinical_information) -> str:

        """Generate a VLM report for the given base64-encoded image data."""
        self.logger.info("Generating VLM report for image")
        
        vlm_agent_prompt = f"""You are a medical visual AI specialized in dermatology and STDs/STIs. You will receive a patient's skin image(s) (from any body part) along with their associated clinical information. Your task is to analyze and diagnose the condition using a structured, multi-step approach that explicitly incorporates deep analytical thinking and reasoning.

        CONTEXT:
        Clinical Information of the patient: {clinical_information}

        1. Image Analysis:
        - Examine the image(s) meticulously.
        - Describe the anatomical location and all notable visual features, including:
            • Color and any changes in color.
            • Size and shape of lesions or rashes.
            • Texture and surface quality (e.g., smooth, rough, scaly, moist).
            • Abnormalities such as rashes, macules/papules, vesicles/blisters, pustules, ulcers/open sores, warts/growths, nodules, discharge, crusting, or erosion.
            • Patterns or distribution if multiple lesions are present (e.g., clustered, linear, generalized, symmetrical).
            • Consider the patient's skin tone in your descriptions.

        2. Clinical Correlation:
        - Integrate the provided patient information (demographics, sexual history, symptoms, duration, medical history).
        - Note relevant factors like age, sex, and risk behaviors (e.g., unprotected sex, new partner, IV drug use) pertinent to STIs.
        - Account for symptoms such as itching, pain, burning, and systemic signs (e.g., fever, malaise) to support or refute potential diagnoses.
        - Summarize how the clinical details align with or contradict the visual findings.

        3. Analytical Reasoning & Conflict Resolution:
        - Deliberately think through and weigh the evidence from both the image analysis and clinical correlation.
        - Identify any discrepancies between the visual findings and clinical history (e.g., a high-confidence "Normal" image output versus clinical signs of pathology).
        - Prioritize clinical history and detailed morphological descriptions over isolated high-confidence image outputs.
        - Document your reasoning process by outlining key points, conflicts identified, and how these conflicts are resolved.

        4. Diagnosis:
        - Based on the image analysis, clinical correlation, and your analytical reasoning, determine the most likely diagnosis.
        - For lesions suggesting STD/STI involvement, consider conditions such as herpes, syphilis, HPV (genital warts), chlamydia/gonococcal infections, HIV-related rash, and molluscum contagiosum.
        - Also, consider non-STI dermatologic conditions (e.g., eczema, psoriasis, fungal infections, contact dermatitis, acne).
        - Select the single best fitting diagnosis as the main diagnosis.
        - Provide a confidence score (0–100) for the main diagnosis along with a clear explanation that highlights the key visual, clinical, and reasoning factors.
        - List alternative plausible diagnoses as differential diagnoses; each must include:
            • The diagnosis name.
            • A confidence score (0–100).
            • A brief explanation for its consideration.

        5. Recommendations:
        - Suggest any confirmatory tests (e.g., PCR, blood tests, biopsy) if needed.
        - Recommend referrals or follow-up (e.g., refer to a dermatologist or appropriate specialist).
        - Advise on immediate management or treatment (e.g., initiate antifungal cream, prescribe oral antibiotics, or advise temporary sexual abstinence).
        - Include preventive recommendations where applicable (e.g., safe sex practices, improved hygiene, routine monitoring).

        6. Normal Case:
        - If the image shows no abnormality:
            "main_diagnosis": "Normal skin (no significant abnormality)"
            "differential_diagnosis": [] (empty list)
            "main_diagnosis_confidence": 100
            "recommendations": "No treatment needed. Maintain routine skincare and monitor."

        7. All Regions & Skin Types:
        - Apply this approach for any body part (e.g., genitals, oral cavity, or elsewhere) and for all skin tones.
        - Adjust differential diagnoses appropriately based on lesion location, patient history, and skin color variations.

        8. Final Review & Diagnosis Finalization:
        - Conduct a final review of all integrated data and your reasoning process.
        - Confirm that any conflicts between image findings and clinical information have been resolved, with a clear priority given to the clinical context and logical reasoning.
        - If uncertainties or conflicts remain, explicitly recommend confirmatory testing to ensure diagnostic accuracy.

        9. Output Format:
        - Respond ONLY with a JSON object containing exactly the following fields (and no additional text):

        ```json
        {{
          "image_analysis": "Detailed description of visual findings.",
          "clinical_correlation": "Summary of how clinical information was integrated with the visual analysis.",
          "main_diagnosis": "Name of the most likely condition",
          "main_diagnosis_confidence": 0-100,
          "main_diagnosis_reasoning": "Explanation of why this diagnosis was chosen, highlighting key visual, clinical, and reasoning factors.",
          "differential_diagnosis": [
            {{
              "diagnosis": "Alternative diagnosis 1",
              "confidence_score": 0-100,
              "reasoning": "Explanation for considering this alternative."
            }},
            {{
              "diagnosis": "Alternative diagnosis 2",
              "confidence_score": 0-100,
              "reasoning": "Explanation for considering this alternative."
            }}
            // Additional differential diagnoses as needed.
          ],
          "recommendations": "Next steps such as tests, referrals, or treatments.",
          "debug_msg": "Any debugging information or notes about the analysis process."
        }}
        ```
        """

        try:
            vlm_analysis = await self.get_o1_completion(
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
        
    async def get_o1_completion(self,
                               prompt: str,
                               model: str = "o1",
                               max_retries: int = 3,
                               output_json_schema=None,
                               reasoning_effort: str = "high",
                               image_data: str = None):
        """
        Get completion from OpenAI's o1 model with image analysis capabilities.
        """
        def is_json(content):
            try:
                json.loads(content)
                return True
            except:
                return False

        for attempt in range(max_retries):
            try:
                self.logger.info(f"Attempting o1 analysis (attempt {attempt + 1}/{max_retries})")
                
                # First message with system/developer instructions
                messages = [
                    {
                        "role": "developer",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ]
                
                # Add a separate user message with the image if provided
                if image_data:
                    messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    })

                completion_params = {
                    "model": model,
                    "messages": messages,
                    "reasoning_effort": reasoning_effort
                }
                
                # Add response format if schema provided
                if output_json_schema:
                    completion_params["response_format"] = {
                        "type": "json_schema",
                        "json_schema": output_json_schema
                    }

                completion = self.client.chat.completions.create(**completion_params)
                content = completion.choices[0].message.content

                if is_json(content):
                    parsed_content = json.loads(content)
                    return parsed_content
                else:
                    self.logger.warning("o1 response not valid JSON. Retrying...")

            except Exception as e:
                self.logger.error(
                    f"Error in get_o1_completion (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )
                if attempt == max_retries - 1:
                    return None
                await asyncio.sleep(2 ** attempt)
        return None
        
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

            Diagnoses can be expressed at different levels of specificity. Sometimes, the ground truth diagnosis is a broad condition (e.g., "HPV"), while the predicted diagnosis refers to a specific manifestation or common clinical presentation of that condition (e.g., "genital warts"). 

            Please consider the following:
            - If the predicted diagnosis is a broader category that sufficiently covers the ground truth condition, return {{ "match": true }}.
            - If the predicted diagnosis is a specific manifestation or subset that is a recognized clinical presentation of the ground truth condition, return {{ "match": true }}.
            - Otherwise, if the predicted diagnosis does not sufficiently capture the ground truth condition, return {{ "match": false }}.

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
            
    async def get_skin_filter_report(self, image_data: str) -> str:
        """
        Generate a skin filter report to validate if the image contains human skin.
        
        Args:
            image_data: Base64 encoded image data or image URL
            
        Returns:
            str: JSON formatted skin filter report
        """
        self.logger.info("Generating skin filter report for image")
        
        try:
            # Get the skin filter analysis using the VLM completion
            skin_analysis = await self.get_vlm_completion(
                prompt=skin_filter_agent_prompt,
                output_json_schema=otiz_skin_filter_schema,
                image_data=image_data
            )
            
            if skin_analysis:
                # Convert the analysis to a formatted string
                skin_report = json.dumps(skin_analysis, indent=2)
                self.logger.info("Successfully generated skin filter report")
                return skin_report
            else:
                self.logger.error("Failed to generate skin filter analysis") 
                return "Error: Unable to analyze image"
                
        except Exception as e:
            self.logger.error(f"Error generating skin filter report: {str(e)}")
            return f"Error: {str(e)}"

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
        image_base64 = base64.b64encode(image_data).decode('utf-8') if image_data else ""
        
        # First check if image contains human skin
        skin_filter_report = await openai_service.get_skin_filter_report(image_base64) if image_data else None
        
        # Default values for reports
        lvrm_diagnosis_report = None
        otiz_report = None
        is_human_skin = True  # Default to True if no image data
        
        if skin_filter_report and "Error" not in skin_filter_report:
            try:
                skin_filter_data = json.loads(skin_filter_report)
                is_human_skin = skin_filter_data.get("isHumanSkin", True)
            except json.JSONDecodeError:
                is_human_skin = True  # Default to True if parsing fails
        
        # Only proceed with analysis if human skin is detected
        if is_human_skin and image_data:
            lvrm_diagnosis_report = await openai_service.get_lvrm_report(image_base64, case_history)
            otiz_report = await openai_service.get_otiz_report(lvrm_diagnosis_report)
        else:
            # If not human skin, set empty reports
            if image_data:  # Only set this for non-human skin (not for missing images)
                lvrm_diagnosis_report = json.dumps({"image_analysis": "Not human skin detected", "main_diagnosis": "Inanimate object"})
                otiz_report = json.dumps({"diagnosis": "Inanimate object", "differential_diagnosis": []})
            else:
                lvrm_diagnosis_report = "No image data for VLM"
                
        return lvrm_diagnosis_report, otiz_report, is_human_skin
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None, True  # Default to True on error

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
        lvrm_diagnosis_report, otiz_report, is_human_skin = asyncio.run(
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

                # If not human skin, we already know it's a mismatch
                if not is_human_skin and image_data:
                    match_indicator = False
                    st.session_state["gender_stats"][gender]["misclassified"] += 1
                else:
                    # Check accuracy for human skin images
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
                            # Only show differential diagnoses if it's human skin
                            if is_human_skin and otiz_data.get('differential_diagnosis'):
                                st.markdown("\n**Differential Diagnoses:**")
                                for dd in otiz_data['differential_diagnosis']:
                                    st.write(f"- {dd['diagnostic_possibility']}: {dd['confidence_score']:.2%}")
                    except:
                        st.write("No valid differential diagnosis found.")
                
                st.markdown("### Analysis")
                if otiz_report and "Error" not in otiz_report and is_human_skin:
                    try:
                        otiz_data = json.loads(otiz_report)
                        out = otiz_data.get("output", {})
                        st.markdown(f"**Reasoning Summary:** {out.get('reasoning_summary', '')}")
                        st.markdown(f"**Response:** {out.get('response', '')}")
                    except:
                        st.write("No valid analysis found.")
                elif not is_human_skin and image_data:
                    st.markdown("**Reasoning Summary:** Image does not contain human skin.")
                    st.markdown("**Response:** Cannot provide medical analysis for non-human skin images.")

            with col3:
                st.markdown("### Reports")
                with st.expander("OTIZ Report", expanded=True):
                    if not is_human_skin and image_data:
                        st.info("No OTIZ Report - Image does not contain human skin")
                    else:
                        st.json(otiz_report if otiz_report else "No OTIZ Report")
                with st.expander("LVRM Diagnosis Report", expanded=True):
                    if not is_human_skin and image_data:
                        st.info("No LVRM Report - Image does not contain human skin")
                    else:
                        st.json(lvrm_diagnosis_report if lvrm_diagnosis_report else "No VLM Report")
            

            st.markdown("---")

if __name__ == "__main__":
    main()