from flask import Flask, request, jsonify
from transformers import AutoTokenizer
import transformers
import torch

app = Flask(__name__)

model = "./my-autotrain-llm"
tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    framework="pt"
)

@app.route('/chat', methods=['POST'])
def chat():

    input_text = request.json.get('input_text', '')
    
    sequences = pipeline(
        input_text,
        do_sample=True, 
        top_p=0.9, 
        temperature=0.7,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=50,
        truncation=True,
    )
    
    for seq in sequences:
        generated_text = seq['generated_text']
        result = generated_text.replace(input_text,"").strip()
        return jsonify({"response": result}, ensure_ascii=False)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)