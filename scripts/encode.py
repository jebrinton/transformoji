import os
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.config import MODEL_IDS

# Precompile regex pattern for emoji extraction (including spaces)
EMOJI_SPACE_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags (iOS)
    "\U00002700-\U000027BF"  # dingbats
    "\U0001f926-\U0001f937"  # gestures
    "\U00010000-\U0010ffff"  # other unicode
    "\u2640-\u2642"  # gender symbols
    "\u2600-\u2B55"  # misc symbols
    "\u200d"  # zero width joiner
    "\u23cf"  # eject symbol
    "\u23e9"  # fast forward
    "\u231a"  # watch
    "\ufe0f"  # variation selector
    "\u3030"  # wavy dash
    "\s"  # space
    "]+",
    flags=re.UNICODE
)

def clean_encoding(response):
    """Clean the model response to extract only emoji strings and spaces"""
    if "Final answer:" not in response:
        logging.warning(f"Response {response} does not contain 'Final answer:'")
        final_answer = response
    else:
        final_answer = response.split("Final answer:")[1].strip()
    
    # Find all emoji and space sequences
    emoji_sequences = EMOJI_SPACE_PATTERN.findall(final_answer)

    # Join them into a single string and strip only leading/trailing whitespace
    clean_result = ''.join(emoji_sequences).strip()

    return clean_result

def generate_encoding(model, tokenizer, sentence, encoder_prompt):
    """Generate emoji encoding for a sentence using the model"""
    # Format the prompt with the sentence
    prompt = encoder_prompt.format(sentence=sentence)

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,  # Adjust as needed
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode response
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    return response

def main(test_mode=False):
    # Load data
    sentences = load_sentences()
    templates = load_templates()
    encoder_prompt = templates['encoder_prompt']

    if test_mode:
        # Use only first model and first 3 sentences for testing
        models_to_process = dict(list(MODEL_IDS.items())[:1])
        sentences = sentences[:3]
        logging.info("TEST MODE: Using only first model and first 3 sentences")
    else:
        models_to_process = MODEL_IDS

    logging.info(f"Loaded {len(sentences)} sentences")
    logging.info(f"Models to process: {list(models_to_process.keys())}")

    # Process each model
    for model_name, model_id in models_to_process.items():
        torch.cuda.empty_cache()
        logging.info(f"\nProcessing model: {model_name}")

        try:
            # Load model and tokenizer
            logging.info("Loading model...")
            tokenizer = AutoTokenizer.from_pretrained(model_id, device_map="auto")
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

            # Store encodings for this model
            encodings = []

            # Process each sentence
            for i, sentence_data in enumerate(sentences):
                logging.info(f"Processing sentence {i+1}/{len(sentences)}: {sentence_data['text'][:50]}...")

                sentence = sentence_data['text']

                # Generate encoding
                raw_response = generate_encoding(model, tokenizer, sentence, encoder_prompt)
                logging.info(f"  Raw response: {raw_response}")

                # Clean encoding
                clean_response = clean_encoding(raw_response)
                logging.info(f"  Clean encoding: '{clean_response}'")

                # Store result
                encoding_data = {
                    'sentence': sentence,
                    'raw_response': raw_response,
                    'clean_encoding': clean_response
                }
                encodings.append(encoding_data)

            # Save encodings to file
            suffix = "_test" if test_mode else ""
            output_file = f'/projectnb/mcnet/jbrin/transformoji/encodings/{model_name}_encodings{suffix}.json'
            with open(output_file, 'w') as f:
                json.dump(encodings, f, indent=2, ensure_ascii=False)

            logging.info(f"Saved encodings to {output_file}")

            del model
            del tokenizer
            torch.cuda.empty_cache()

        except Exception as e:
            logging.info(f"Error processing model {model_name}: {e}")
            continue

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    torch.set_float32_matmul_precision('high')

    test_mode = "--test" in sys.argv
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        logging.info("DRY RUN MODE: Testing data loading and logic without loading models")

        # Test data loading
        sentences = load_sentences()
        templates = load_templates()
        encoder_prompt = templates['encoder_prompt']

        logging.info(f"Successfully loaded {len(sentences)} sentences")
        logging.info(f"First sentence: {sentences[0]['text']}")

        # fill in encoder prompt template with first sentence
        encoder_prompt = encoder_prompt.format(sentence=sentences[0]['text'])
        logging.info(f"Encoder prompt: {encoder_prompt}")

        # Test emoji cleaning
        test_response = "Here are some emojis: 🐨🏃‍♂️💨 for the kangaroo jumping!"
        clean_result = clean_encoding(test_response)
        logging.info(f"Test cleaning: '{test_response}' -> '{clean_result}'")

        logging.info(f"Models available: {list(MODEL_IDS.keys())}")

        # Show what would be processed
        if test_mode:
            models_to_process = dict(list(MODEL_IDS.items())[:1])
            sentences_subset = sentences[:3]
        else:
            models_to_process = MODEL_IDS
            sentences_subset = sentences

        logging.info(f"Would process {len(models_to_process)} models and {len(sentences_subset)} sentences")
        logging.info("Dry run complete!")

    else:
        main(test_mode=test_mode)
