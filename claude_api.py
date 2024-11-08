from anthropic import Anthropic
import json

# The prompt that insights should be generated from.
prompt = "a happy cat eating an orange and an apple with two trees behind it"

client = Anthropic(
    api_key="",
)

message = client.messages.create(
    max_tokens=150,
    messages=[
        {
            "role": "user",
            "content": "I will give you a short prompt that is used to generate an image. I want you to answer, in JSON format, whether the following attributes of an image are specified clearly (true) or left ambiguous/open to interpretation in the prompt (false): objects, colors, emotions, spatial relations and activities. For each category, generate a short description of what the prompt wants IF and ONLY IF the category is specified. If it is ambiguous, leave that section blank. For the objects section, if you are generating a description, generate it only with the necessary objects comma separated. Return ONLY in JSON format. Prompt: " + prompt,
        }
    ],
    model="claude-3-opus-20240229",
)

data = json.loads(message.content[0].text)

with open('response.json', 'w') as f:
    json.dump(data, f, indent=2)

print("JSON file 'output.json' created successfully.")

