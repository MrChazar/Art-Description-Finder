

def create_prompt_func(text):
    prompt = f"""Human:
    Task: Categorize the text's emotional tone as either 'neutral or no emotion' or identify the presence of one or more of the given emotions (anger, anticipation, disgust, fear, joy, love, optimism, pessimism, sadness, surprise, trust).
    Text: {text}
    This text contains emotions:
    
    Assistant:
    """
    return prompt



