## Imports
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from utils import *
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import re
import uvicorn
from datetime import datetime



import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

def build_notes_index(root_folder):
    """
    Walks through root_folder and finds all .txt files.
    Returns a dict: { filename: path }.
    If duplicate filenames are found, stores list of paths.
    """
    notes_index = {}

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for fname in filenames:
            if fname.lower().endswith('.txt') or fname.lower().endswith('.xlsx'):
                # build relative path
                full_path = os.path.join(dirpath, fname)
                # Make path relative to root_folder if you prefer
                rel_path = os.path.relpath(full_path, start=root_folder)
                # Optionally prefix with ./notes/
                dict_value = os.path.join(root_folder, rel_path)

                # handle possible duplicate filenames
                if fname in notes_index:
                    # if existing value is a str, convert to list
                    existing = notes_index[fname]
                    if isinstance(existing, str):
                        notes_index[fname] = [ existing ]
                    # now append new path
                    notes_index[fname].append(dict_value)
                else:
                    notes_index[fname] = dict_value

    return notes_index

def get_llm_response(
    user_prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str = "gpt-4o-mini",
    get_tokens: bool = False
):
    """
    Calls the OpenAI Chat Completions API with specified prompts and model.

    Args:
        user_prompt (str): The user's question or instruction.
        system_prompt (str): The system message to set the assistant's behavior.
        model (str): The name of the GPT model to use (e.g., 'gpt-4o-mini', 'gpt-3.5-turbo').
        get_tokens (bool): Whether to return token usage details.

    Returns:
        - If get_tokens=False: str (assistant's response)
        - If get_tokens=True: dict with keys {"response", "tokens"}
          where tokens = {"input_tokens", "output_tokens", "total_tokens"}
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        # API Call
        response = client.chat.completions.create(
            model=model,
            messages=messages,
        )

        # Extract message
        message_content = (
            response.choices[0].message.content
            if response.choices and response.choices[0].message
            else "Error: Empty response from API."
        )

        # Extract token usage if available
        usage = getattr(response, "usage", None)
        tokens_dict = {
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        } if usage else None

        if get_tokens:
            return {
                "response": message_content,
                "tokens": tokens_dict
            }
        else:
            return message_content

    # --- Error Handling ---
    except AuthenticationError:
        return "API Error: Authentication failed. Check your API key."
    except RateLimitError:
        return "API Error: Rate limit exceeded. Please wait before retrying."
    except APIError as e:
        return f"API Error: An OpenAI API error occurred: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def jsonify_output(llm_output: str):
    """
    Safely converts a stringified LLM response into a Python dictionary.
    Designed for cases where the LLM is instructed to return JSON that can be parsed by json.loads(),
    but may include extra formatting, markdown, or escaped characters.
    
    Args:
        llm_output (str): The raw LLM string output.
    
    Returns:
        dict: Parsed dictionary with keys/values from the LLM output.
    """
    # Step 1: Clean common wrappers (e.g., code blocks, quotes)
    cleaned = llm_output.strip()
    
    # Remove markdown code block syntax (```json ... ```)
    cleaned = re.sub(r"^```(json)?", "", cleaned)
    cleaned = re.sub(r"```$", "", cleaned)
    
    # Step 2: Handle single quotes or extra escaping
    # Convert Python-style dict string → JSON-compatible
    cleaned = cleaned.strip()
    
    # If it starts and ends with quotes (common case), remove them
    if cleaned.startswith("'") and cleaned.endswith("'"):
        cleaned = cleaned[1:-1]
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1]

    # Replace any escaped quotes \" or \'
    cleaned = cleaned.replace('\\"', '"').replace("\\'", "'")

    # Step 3: Try loading directly as JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print("⚠️ json.loads() failed, trying fallback fix…")
        # Attempt to extract JSON content using regex
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception as inner_e:
                print("❌ Still failed to parse JSON:", inner_e)
        raise ValueError(f"Failed to parse JSON from LLM output: {e}\n\nCleaned text:\n{cleaned[:500]}")

def modify_week(df: pd.DataFrame, muscle_areas: list[str]) -> dict:
    """
    Increase the 'week' counter by 1 for all exercises in the given muscle area(s).
    
    Args:
        df: DataFrame containing the tracking table.
        muscle_areas: List of muscle area names (e.g., ["Chest", "Back"]).
        
    Returns:
        dict with status, list of muscle_areas updated, and rows_updated per area.
    """
    results = []
    for area in muscle_areas:
        mask = df["muscle area"].str.lower() == area.lower()
        if not mask.any():
            results.append({"muscle_area": area, "status": "error", "message": "No exercises found for this muscle area"})
        else:
            df.loc[mask, "week"] += 1
            new_week = int(df.loc[mask, "week"].iloc[0])
            results.append({"muscle_area": area, "status": "success", "new_week": new_week, "rows_updated": int(mask.sum())})
    return {"results": results}

def modify_weight(df: pd.DataFrame, updates: list[dict]) -> dict:
    """
    Update the 'max weight' for given exercise(s) if the new weight is greater than the current recorded weight.
    
    Args:
        df: DataFrame containing the tracking table.
        updates: List of update dicts, each with keys:
                 muscle_area (str), exercise_name (str), new_weight (float).
    
    Returns:
        dict with results list showing status, old_weight, new_weight for each update.
    """
    results = []
    for upd in updates:
        area = upd.get("muscle_area")
        exer = upd.get("exercise_name")
        new_w = upd.get("new_weight")
        mask = (df["muscle area"].str.lower() == area.lower()) & \
               (df["exercise name"].str.lower() == exer.lower())
        if not mask.any():
            results.append({
                "muscle_area": area,
                "exercise_name": exer,
                "status": "error",
                "message": "Exercise not found for this muscle area"
            })
        else:
            current_weight = float(df.loc[mask, "max weight"].iloc[0])
            if new_w > current_weight:
                df.loc[mask, "max weight"] = new_w
                df.loc[mask, "week"] = 1  # reset week for that exercise
                results.append({
                    "muscle_area": area,
                    "exercise_name": exer,
                    "status": "success",
                    "old_weight": current_weight,
                    "new_weight": new_w
                })
            else:
                results.append({
                    "muscle_area": area,
                    "exercise_name": exer,
                    "status": "no_change",
                    "old_weight": current_weight,
                    "attempted_weight": new_w,
                    "message": "New weight is not greater than current recorded weight"
                })
    return {"results": results}

def update_calorie_tracker(items: list[dict] , df: pd.DataFrame = None):
    """
    Appends multiple calorie entries into the calorie tracking Excel file.

    items format (coming from LLM JSON):
    [
        { "item": "Boiled eggs, 2 large", "calories": 155.0, "protein": 12.6 },
        { "item": "Soya chunks (dry), 20 g", "calories": 73.2, "protein": 10.2 }
    ]
    """

    # ✅ Use today's date
    today = datetime.now().strftime("%Y-%m-%d")
    rows_added = []
    for entry in items:
        new_row = {
            "date": today,
            "item": entry["item"],
            "calories": float(entry["calories"]),
            "protein": float(entry["protein"]),
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        rows_added.append(new_row)

    return {
        "status": "success",
        "final_df": df,
        "rows_added": rows_added,
    }


