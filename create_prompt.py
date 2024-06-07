import glob
from pathlib import Path
from deep_translator import GoogleTranslator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


videosPathes=glob.glob("videos/*")
textPathes=glob.glob("text_from_videos/*_text.txt")
langPathes=glob.glob("text_from_videos/*_lang.txt")

def create_pipeline():
    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-128k-instruct",
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
        load_in_4bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    return pipe, generation_args

def translator(list_of_strings, TRANSLATE_FROM='de'):
    return [GoogleTranslator(source=TRANSLATE_FROM, target='en').translate(string) for string in list_of_strings]

def extract_keywords(messages, pipeline, GENERATION_ARGS):
    return pipeline(messages, **GENERATION_ARGS)[0]['generated_text']

def read_text(filepath, number_of_lines=40):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        chunks = [lines[i:i + number_of_lines] for i in range(0, len(lines), number_of_lines)]
        chunks = [''.join(chunk) for chunk in chunks]
        f.close()
    return chunks



def create_prompt_for_SD(PIPELINE, GENERATION_ARGS, lines_in_chunk=30):
    for i in range(len(textPathes)):

        text_chunks=read_text(textPathes[i], number_of_lines=lines_in_chunk)

        if len(langPathes)>0:
            with open(langPathes[i],'r') as f:
                text_language=f.read()
                f.close()
            if text_language != 'en':
                text_chunks= translator(text_chunks, text_language)


        all_text=''
        for chunk in text_chunks:
            messages = [
                {"role": "user", "content": "extract 4 main keywords: The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."},
                {"role": "assistant", "content": "Paris, tower, the tallest man-made structure, France"},
                {"role": "user", "content": "extract 4 main keywords: In the quiet town of Elmswood, nestled between rolling hills and ancient forests, there existed a small bookstore that was unlike any other. 'Whispers of Time,' it was called, and it was renowned for its collection of rare and magical books. The owner, Mr. Thorne, was an enigmatic figure with a deep knowledge of the mystical arts. His store was a sanctuary for those seeking wisdom and adventure beyond the ordinary world. One rainy afternoon, a young woman named Clara stumbled into the bookstore, escaping the storm. She was immediately captivated by the scent of old paper and the soft glow of lanterns illuminating the shelves. As she wandered through the aisles, a particular book caught her eye. Its cover was intricately decorated with symbols she couldn't decipher, and it seemed to hum with an energy all its own."},
                {"role": "assistant", "content": "quiet town, bookstore, rainy afternoon, book "},
                {"role": "user", "content": f"extract at least 4 main keywords: {chunk}"},
            ]

            all_text = all_text+extract_keywords(messages, PIPELINE, GENERATION_ARGS)+', '

        messages = [
            {"role": "user", "content": "extract at least 4 main keywords: The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."},
            {"role": "assistant", "content": "Paris, tower, the tallest man-made structure, France"},
            {"role": "user", "content": "extract 4 main keywords: In the quiet town of Elmswood, nestled between rolling hills and ancient forests, there existed a small bookstore that was unlike any other. 'Whispers of Time,' it was called, and it was renowned for its collection of rare and magical books. The owner, Mr. Thorne, was an enigmatic figure with a deep knowledge of the mystical arts. His store was a sanctuary for those seeking wisdom and adventure beyond the ordinary world. One rainy afternoon, a young woman named Clara stumbled into the bookstore, escaping the storm. She was immediately captivated by the scent of old paper and the soft glow of lanterns illuminating the shelves. As she wandered through the aisles, a particular book caught her eye. Its cover was intricately decorated with symbols she couldn't decipher, and it seemed to hum with an energy all its own."},
            {"role": "assistant", "content": "quiet town, bookstore, rainy afternoon, book"},
            {"role": "user", "content": f"extract at least 4 main keywords: {all_text}"},
        ]
        # output = PIPELINE(messages, **GENERATION_ARGS)
        final_result=' '.join(extract_keywords(messages, PIPELINE, GENERATION_ARGS).split()[:15])

        with open(f'generated_prompt_for_SD/{Path(videosPathes[i]).stem}_prompt.txt','w') as f:
            f.write(final_result)
            f.close()




# torch.random.manual_seed(0)
# PIPELINE, GENERATION_ARGS =create_pipeline()
# create_prompt_for_SD(PIPELINE, GENERATION_ARGS,lines_in_chunk=15)

