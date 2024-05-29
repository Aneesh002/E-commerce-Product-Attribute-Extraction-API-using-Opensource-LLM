from fastapi import FastAPI, Body
from pydantic import BaseModel
import uvicorn
import os

from model import initialize_tokenizer, load_model
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

model_name = os.getenv("MODEL_NAME")
hf_token = os.getenv("HUGGINGFACE_TOKEN")

# Initialize tokenizer and model
tokenizer = initialize_tokenizer(model_name, hf_token)
model = load_model(model_name, hf_token)
stop_token_ids = [0]


class HTMLContent(BaseModel):
    html: str


@app.post("/extract")
async def extract_attributes(content: HTMLContent):
    html_text = content.html

    output_structure = {
        'product_name': {
            'value': None,
            'selector': None
        },
        'product_brand': {
            'value': None,
            'selector': None
        },
        'price': {
            'value': None,
            'selector': None
        },
        'discount': {
            'value': None,
            'selector': None
        },
        'description': {
            'value': None,
            'selector': None
        },
        'specifications': {
            'value': None,
            'selector': None
        },
        'images': {
            'value': None,
            'selector': None
        },
        'reviews': {
            'value': None,
            'selector': None
        }
    }

    chat = [
        {
            "role": "user",
            "content": f"Extract meaningful information from the following HTML content \n\n {html_text} and provide it in JSON format as specified: {output_structure}"
        },
    ]

    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
    outputs = model.generate(input_ids=inputs, max_new_tokens=4000)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(text)

    #This portion is the Post processing after the llm output in required json
    json_start = text.find("[/INST]")
    json_text = text[json_start + len("[/INST]"):].strip()

    print(json_text)

    return json_text


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
