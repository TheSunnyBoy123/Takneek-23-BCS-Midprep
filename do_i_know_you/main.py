from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("qu-bit/SuperLLM")
model = AutoModelForCausalLM.from_pretrained("qu-bit/SuperLLM")

input_text = "List all RAW agents"

input_ids = tokenizer(input_text, return_tensors='pt').input_ids

output = model.generate(input_ids, max_length=100, num_return_sequences=1)

decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)

print(decoded_output)