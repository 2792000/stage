import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
def generateIA(pathToJson):

    # Load the model and tokenizer
    model_name = "Mahalingam/DistilBart-Med-Summary"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Read the transcription from a JSON file
    json_file_path = pathToJson

    # Load the JSON file
    with open(json_file_path, "r") as file:
        data = json.load(file)

    # Extract the transcription text
    #transcription = data["transcription"]
    transcription="hello Mrs Smith how are you feeling today good morning I have been feeling a little under the weather lately I am sorry to hear that can you describe your symptoms for me yes I have had a persistent headache for the past few days it starts in the morning and sometimes it lasts all day I see do you have any other symptoms like nausea or dizziness yes I feel a little more period"
    # Prepare the input for the model
    inputs = tokenizer(transcription, return_tensors="pt", truncation=True, padding="longest")

    # Generate the summary
    outputs = model.generate(inputs["input_ids"], max_length=600, num_beams=4, early_stopping=True)

    # Decode and print the summary, explicitly setting clean_up_tokenization_spaces
    generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

    print("Generated Summary:")
    print(generated_summary)
    return generated_summary
