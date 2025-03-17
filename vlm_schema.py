vlm_schema = {
  "name": "clinical_analysis_report",
  "strict": True,
  "schema": {
    "type": "object",
    "properties": {
      "image_analysis": {
        "type": "string",
        "description": "Detailed description of visual findings."
      },
      "clinical_correlation": {
        "type": "string",
        "description": "Summary of how clinical information was integrated with the visual analysis."
      },
      "main_diagnosis": {
        "type": "string",
        "description": "Name of the most likely condition."
      },
      "main_diagnosis_confidence": {
        "type": "number",
        "description": "Confidence level of the main diagnosis, ranging from 0 to 100."
      },
      "main_diagnosis_reasoning": {
        "type": "string",
        "description": "Explanation of why this diagnosis was chosen, highlighting key visual and clinical factors."
      },
      "differential_diagnosis": {
        "type": "array",
        "description": "List of alternative diagnoses considered.",
        "items": {
          "type": "object",
          "properties": {
            "diagnosis": {
              "type": "string",
              "description": "Alternative diagnosis."
            },
            "confidence_score": {
              "type": "number",
              "description": "Confidence score for the alternative diagnosis, ranging from 0 to 100."
            },
            "reasoning": {
              "type": "string",
              "description": "Explanation for considering this alternative."
            }
          },
          "required": [
            "diagnosis",
            "confidence_score",
            "reasoning"
          ],
          "additionalProperties": False
        }
      },
      "recommendations": {
        "type": "string",
        "description": "Next steps such as tests, referrals, or treatments."
      },
      "debug_msg": {
        "type": "string",
        "description": "Response related to debugging."
      }
    },
    "required": [
      "image_analysis",
      "clinical_correlation",
      "main_diagnosis",
      "main_diagnosis_confidence",
      "main_diagnosis_reasoning",
      "differential_diagnosis",
      "recommendations",
      "debug_msg"
    ],
    "additionalProperties": False
  }
}