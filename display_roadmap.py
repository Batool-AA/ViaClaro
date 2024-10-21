import openai

# Set your OpenAI API key
openai.api_key = ''

def generate_roadmap(domain):
    prompt = f"Create a comprehensive roadmap for ${domain}. The roadmap should contain all the key skills to excel in this profession. Also mention the dependencies between these skills."
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or "gpt-4" if you have access
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    roadmap = response['choices'][0]['message']['content']
    return roadmap



