prog_over_tools = \
[
  {
    "type": "function",
    "name": "modify_week",
    "description": "Increase the week counter by 1 for one or more muscle areas.",
    "parameters": {
      "type": "object",
      "properties": {
        "muscle_areas": {
          "type": "array",
          "items": {
            "type": "string",
            "description": "Name of a muscle area (e.g., 'Chest', 'Back', 'Legs')."
          },
          "description": "List of muscle areas whose week counter should be incremented."
        }
      },
      "required": ["muscle_areas"]
    }
  },
  {
    "type": "function",
    "name": "modify_weight",
    "description": "Update the max weight for one or more specific exercises; resets week for each exercise when weight increases.",
    "parameters": {
      "type": "object",
      "properties": {
        "updates": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "muscle_area": {
                "type": "string",
                "description": "The muscle area for the exercise."
              },
              "exercise_name": {
                "type": "string",
                "description": "The exact exercise name to update (e.g., 'Incline Dumbbell Chest Press')."
              },
              "new_weight": {
                "type": "number",
                "description": "The new weight lifted (in kg or chosen unit)."
              }
            },
            "required": ["muscle_area", "exercise_name", "new_weight"]
          },
          "description": "List of exercises to update with their new weights."
        }
      },
      "required": ["updates"]
    }
  }
]
