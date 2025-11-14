## Imports
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from utils import *
from tools import *
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Any, Dict, Optional
import uvicorn
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os

JSON_KEYFILE = './service-account-creds.json'
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name(JSON_KEYFILE, scope)
gc_client = gspread.authorize(creds)
spreadsheets = gc_client.openall()


load_dotenv(dotenv_path="../.env")
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))

async def process_user_input(user_input: str) -> Dict[str, Any]:
    if not user_input:
        raise HTTPException(status_code=400, detail="Empty input not allowed.")

    # region Step 1: Getting Intent
    notes_path_dict = build_notes_index("./notes")
    notes_list = list(notes_path_dict.values())

    print("notes path dict: " ,notes_path_dict)
    print(f"Notes found: {len(notes_list)}")


    intnt_prmpt_file = "./prompts/get_intent.txt"
    with open(intnt_prmpt_file, "r", encoding="utf-8") as f:
        intnt_prmpt = f.read()

    notes_desc_path = "./note_descriptions.json"
    with open(notes_desc_path, "r", encoding="utf-8") as f:
        note_desc = json.load(f)

    intnt_response = get_llm_response(
        user_prompt=intnt_prmpt.format(user_query = user_input , note_list = notes_list, note_desc = json.dumps(note_desc)),
        model="gpt-4o-mini",
        get_tokens=True
    )

    try:
        intent_data = jsonify_output(intnt_response["response"])

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM response parsing error: {e}")
    # endregion

    # region Step 2: Performing Action Based on Intent
    action_reasoning = intent_data.get('reasoning', 'No reasoning provided.')
    action_intent = intent_data['intent']
    action_note = intent_data['note']
    print("Determined Intent: ", intent_data['intent'] , "\n Note: ", intent_data['note'], " \n Reasoning: ", action_reasoning)

    if action_note == "no_match":
        raise HTTPException(status_code=400, detail="No matching note found for the given input.")

    elif intent_data['intent'] not in ["retrieve_info", "edit_note"]:
        raise HTTPException(status_code=400, detail="Invalid intent detected.")

    elif intent_data['intent'] == "retrieve_info":
        action_prompt_fname = os.path.join("./prompts", f"{action_intent}.txt")
        with open(action_prompt_fname, "r") as f:
            action_prompt_template = f.read()

        action_note_fname = notes_path_dict.get(action_note, None)
        with open(action_note_fname, "r") as f:
            action_note_content = f.read()

        input_list = [
            {
                "role": "user",
                "content": action_prompt_template.format(
                    user_query=user_input,
                    note_content=action_note_content,
                )
            }
        ]
        # 2. Prompt the model with tools defined
        action_response = client.responses.create(
            model="gpt-5",
            tools=[
                {
                "type": "web_search",
                "user_location": {"type": "approximate"},
                "search_context_size": "low"
                },
            ],
            input=input_list,
            tool_choice="auto",
            store=False,
            reasoning={"effort": "low"},
        )

        try:
            final_output = jsonify_output(action_response.output_text)['final_text']
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM action response parsing error: {e}")

        print("Final Output: \n", final_output)

    elif intent_data['intent'] == "edit_note":
        if intent_data['note'] == "progressive_overload_tracking.xlsx":
            # Handle progressive overload tracking update
            sheet = gc_client.open("progressive_overload_tracking").sheet1
            df = pd.DataFrame(sheet.get_all_records())
            exercise_table_str = df[['muscle area', 'exercise name']].to_string(index=False)

            prog_over_prmpt_file = "./prompts/progressive_overloading_tracking.txt"
            with open(prog_over_prmpt_file, "r", encoding="utf-8") as f:
                PROG_OVER_PROMPT = f.read()

            input_list = [
                {
                    "role": "user",
                    "content": PROG_OVER_PROMPT.format(
                        user_input=user_input,
                        exercise_table=exercise_table_str
                    )
                }
            ]
            # 2. Prompt the model with tools defined
            response = client.responses.create(
                model="gpt-5",
                tools=prog_over_tools,
                input=input_list,
                tool_choice="auto"
                
            )
            for item in response.output:
                if item.type == "function_call":
                    print(item.name)
                    if item.name == "modify_week":
                        args = jsonify_output(item.arguments)['muscle_areas']
                        modify_week(df, args)
                    elif item.name == "modify_weight":
                        args = jsonify_output(item.arguments)['updates']
                        modify_weight(df, args)
                    else:
                        print("Unknown function:", item.name)
            # Save back to Google Sheets
            sheet.clear()
            sheet.update([df.columns.values.tolist()] + df.values.tolist())

        elif intent_data['note'] == "calorie_tracking.xlsx":
            # Handle calorie tracking update
            sheet = gc_client.open("calorie_tracking").sheet1
            df = pd.DataFrame(sheet.get_all_records())
            
            calorie_tracker_path = "./prompts/calorie_tracking.txt"
            with open(calorie_tracker_path, "r", encoding="utf-8") as f:
                CLR_TRCK_PROMPT = f.read()

            input_list = [
                {
                    "role": "user",
                    "content": CLR_TRCK_PROMPT.format(
                        user_input=user_input
                    )
                }
            ]
            # 2. Prompt the model with tools defined
            response = client.responses.create(
                model="gpt-5",
                tools=[
                    {
                    "type": "web_search",
                    "user_location": {"type": "approximate"},
                    "search_context_size": "low"
                    },
                ],
                input=input_list,
                tool_choice="auto",
                store=False,
                reasoning={"effort": "low"},
            )    

            result = jsonify_output(response.output_text)
            items = result["items"]
            update_status = update_calorie_tracker(items, df)
            df = update_status["final_df"]

            sheet.clear()
            sheet.update([df.columns.values.tolist()] + df.values.tolist())
            # print(update_status)

    return {
        "status": "success",
        "data" : {"note_updated": action_note, "intent": action_intent, "error": None}
    }