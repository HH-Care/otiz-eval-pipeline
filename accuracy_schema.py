accuracy_schema = {
  "name": "diagnosis_comparison",
  "strict": True,
  "schema": {
    "type": "object",
    "properties": {
      "match": {
        "type": "boolean",
        "description": "Indicates if the predicted label matches the ground truth accurately."
      }
    },
    "required": [
      "match"
    ],
    "additionalProperties": False
  }
}