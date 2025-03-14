diagnose_schema = {
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