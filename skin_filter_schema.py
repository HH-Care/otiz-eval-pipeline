otiz_skin_filter_schema = {
  "name": "image_skin_analysis",
  "strict": True,
  "schema": {
    "type": "object",
    "properties": {
      "isHumanSkin": {
        "type": "boolean", 
        "description": "Indicates whether human skin is present in the analyzed image."
      },
      "message": {
        "type": "string",
        "description": "A message regarding the outcome of the analysis."
      }
    },
    "required": [
      "isHumanSkin",
      "message"
    ],
    "additionalProperties": False
  }
}