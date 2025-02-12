vlm_schema = {
  "name": "clinical_visual_findings",
  "strict": True,
  "schema": {
    "type": "object",
    "required": [
      "visualFindings",
      "differentialDiagnoses",
      "visualAssessmentSummary"
    ],
    "properties": {
      "visualFindings": {
        "type": "object",
        "required": [
          "lesionDetails",
          "overallDescription"
        ],
        "properties": {
          "lesionDetails": {
            "type": "array",
            "items": {
              "type": "object",
              "required": [
                "lesionType",
                "count",
                "arrangement",
                "size",
                "colorPattern",
                "borders",
                "surfaceChanges",
                "exudateOrDischarge",
                "depthOrElevation",
                "surroundingSkinChanges",
                "patternOrShape",
                "additionalNotes"
              ],
              "properties": {
                "size": {
                  "type": "string",
                  "description": "Size of the lesions (e.g., 'approx. 5 mm in diameter')"
                },
                "count": {
                  "type": "string",
                  "description": "Count of lesions (e.g., 'multiple, more than 10')"
                },
                "borders": {
                  "type": "string",
                  "description": "Borders of the lesions (e.g., 'well-defined', 'irregular and raised')"
                },
                "lesionType": {
                  "type": "string",
                  "description": "The type of lesion (e.g., 'papule', 'plaque', 'vesicle', 'pustule', 'ulcer')"
                },
                "arrangement": {
                  "type": "string",
                  "description": "How the lesions are arranged (e.g., 'clustered', 'grouped', 'linear')"
                },
                "colorPattern": {
                  "type": "string",
                  "description": "Color pattern of the lesions (e.g., 'uniformly red', 'brown center with pale halo')"
                },
                "patternOrShape": {
                  "type": "string",
                  "description": "Pattern or shape of the lesions (e.g., 'annular', 'targetoid', 'serpiginous')"
                },
                "surfaceChanges": {
                  "type": "string",
                  "description": "Changes in the surface of the lesions (e.g., 'scaly', 'crusted', 'smooth', 'lichenified')"
                },
                "additionalNotes": {
                  "type": "string",
                  "description": "Any other notable features (e.g., 'central dimpling', 'blanching on pressure')"
                },
                "depthOrElevation": {
                  "type": "string",
                  "description": "Depth or elevation of the lesions (e.g., 'raised', 'pedunculated', 'flat', 'sunken')"
                },
                "exudateOrDischarge": {
                  "type": "string",
                  "description": "Nature of any exudate or discharge (e.g., 'serous', 'purulent', 'bloody', 'none')"
                },
                "surroundingSkinChanges": {
                  "type": "string",
                  "description": "Changes in the surrounding skin (e.g., 'red halo', 'edema', 'hypopigmentation')"
                }
              },
              "additionalProperties": False
            },
            "description": "List of details about visible lesions."
          },
          "overallDescription": {
            "type": "string",
            "description": "Overall description of the visual findings (e.g., 'Multiple erythematous, well-demarcated papules in a linear arrangement with some crusting')"
          }
        },
        "additionalProperties": False
      },
      "differentialDiagnoses": {
        "type": "array",
        "items": {
          "type": "object",
          "required": [
            "condition",
            "confidenceLevel",
            "visualIndicators"
          ],
          "properties": {
            "condition": {
              "type": "string",
              "description": "Condition being considered (e.g., 'Tinea Corporis', 'Secondary Syphilis', 'Herpes Zoster')"
            },
            "confidenceLevel": {
              "type": "string",
              "description": "Confidence level of the diagnosis (e.g., 'low', 'moderate', 'high' or percentage)"
            },
            "visualIndicators": {
              "type": "array",
              "items": {
                "type": "string",
                "description": "Visual indicator (e.g., 'annular lesion with raised scaly border')"
              },
              "description": "List of visual indicators for the condition."
            }
          },
          "additionalProperties": False
        },
        "description": "List of differential diagnoses with associated information."
      },
      "visualAssessmentSummary": {
        "type": "string",
        "description": "Concise synthesis of visual elements guiding differential diagnoses (e.g., 'The presence of ring-shaped lesions with scaly borders suggests a fungal etiology')"
      }
    },
    "additionalProperties": False
  }
}