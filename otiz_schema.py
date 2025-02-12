otiz_schema ={
  "name": "Dr_Otiz",
  "strict": True,
  "schema": {
    "type": "object",
    "properties": {
      "demographic_data": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string",
            "description": "The name of the patient. If not present, politely ask the patient."
          },
          "age": {
            "type": "number",
            "description": "The age of the patient. If not present, politely ask the patient."
          },
          "gender": {
            "type": "string",
            "description": "The gender of the patient. If not present, politely ask the patient."
          },
          "sexual_health_history": {
            "type": "string",
            "description": "Relevant sexual health issues and previous diagnoses. If not provided, politely ask the patient."
          }
        },
        "required": [
          "name",
          "age",
          "gender",
          "sexual_health_history"
        ],
        "additionalProperties": False
      },
      "primary_complaint": {
        "type": "string",
        "description": "The main health concern of the patient."
      },
      "symptoms": {
        "type": "array",
        "description": "List of symptoms provided by the patient.",
        "items": {
          "type": "string"
        }
      },
      "differential_diagnosis": {
        "type": "array",
        "description": "Possible causes for the symptoms presented.",
        "items": {
          "type": "object",
          "properties": {
            "diagnostic_possibility": {
              "type": "string",
              "description": "A potential diagnosis."
            },
            "confidence_score": {
              "type": "number",
              "description": "Confidence in this diagnosis (0 to 1)."
            }
          },
          "required": [
            "diagnostic_possibility",
            "confidence_score"
          ],
          "additionalProperties": False
        }
      },
      "follow_up_question": {
        "type": "object",
        "description": "A targeted follow-up question.",
        "properties": {
          "question": {
            "type": "string",
            "description": "The follow-up question to ask the patient."
          },
          "impact_score": {
            "type": "number",
            "description": "Impact of this question on refining the diagnosis (0 to 1)."
          }
        },
        "required": [
          "question",
          "impact_score"
        ],
        "additionalProperties": False
      },
      "image_request": {
        "type": "boolean",
        "description": "Indicates if an image of the affected area has been requested."
      },
      "image_analysis_report": {
        "type": "string",
        "description": "Analysis of the uploaded image, including likelihoods of each disease."
      },
      "diagnosis": {
        "type": "string",
        "description": "The most likely diagnosis based on final reasoning based on all information.Strictly provide a midical term, not a description."
      },
      "advice": {
        "type": "string",
        "description": "Professional recommendations and next steps for the patient."
      },
      "presynthesis": {
        "type": "string",
        "description": "Final thought process before giving the response."
      },
      "output": {
        "type": "object",
        "properties": {
          "reasoning_summary": {
            "type": "string",
            "description": "Summary of the assistant's reasoning."
          },
          "confidence_level": {
            "type": "number",
            "description": "Overall confidence level in the diagnosis (0 to 1)."
          },
          "response": {
            "type": "string",
            "description": "Final response to the patient, ensuring a respectful and empathetic tone.Always keep response suit to the patient"
          }
        },
        "required": [
          "reasoning_summary",
          "confidence_level",
          "response"
        ],
        "additionalProperties": False
      }
    },
    "required": [
      "demographic_data",
      "primary_complaint",
      "symptoms",
      "differential_diagnosis",
      "follow_up_question",
      "image_request",
      "image_analysis_report",
      "diagnosis",
      "advice",
      "presynthesis",
      "output"
    ],
    "additionalProperties": False
  }
}