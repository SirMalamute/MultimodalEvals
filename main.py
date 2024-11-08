from image_similarity_measures.evaluate import evaluation
from PIL import Image
import os
from clarifai.client.model import Model
import pickle
from anthropic import Anthropic
import json
from src.colors.color_extractor import main_color_extract
from src.colors.color_results_analyzer import translate_colors
from src.img.image_quality import TechnicalImageQualityAnalyzer
from src.img.image_results_analyzer import ImageQualityBenchmarker
import torch
import clip

# The file to run benchmark on
file_name = "test1"
prompt = "a happy cat eating an orange and an apple with two trees behind it"


def image_quality_test():
    analyzer = TechnicalImageQualityAnalyzer()
    benchmarker = ImageQualityBenchmarker()
    results = [[file_name,analyzer.analyze_all(os.path.join("static", file_name+".jpg"))]]
    analysis, detailed_results = benchmarker.analyze_batch(
        results
    )

    print("Image Quality Analysis:")
    print("Failed Tests: " + str(analysis["failed"]))
    print("Warnings: " + str(analysis["warnings"]))
    print("Status: " + str(analysis['worst_images'][0]['status']))
    #print(json.dumps(analysis, indent=2))

    # print(results)

image_quality_test()

# Let's identify whether to use background image or not.
def background_image_similarity_check():
    print("-------------------------------------------------------------------")
    print("RUNNING WITH/WITHOUT BACKGROUND IMAGE SIMILARITY CHECK:")
    # Load the original and without background images
    org_img_path = os.path.join("static", file_name+".jpg")
    pred_img_path = os.path.join("without_background", file_name+".jpg")

    org_img = Image.open(org_img_path)
    pred_img = Image.open(pred_img_path)

    # Resize the without background image to match the original image dimensions
    pred_img = pred_img.resize(org_img.size)

    # Save the background removed + resized image
    pred_img.save(os.path.join("without_background", file_name+".png"))

    # Evaluate the image similarity
    results = evaluation(
        org_img_path=org_img_path,
        pred_img_path=os.path.join("without_background", file_name+".png"),
        metrics=["rmse", "ssim", "fsim"]
    )

    # Define thresholds for image similarity metrics
    rmse_threshold = 0.05
    ssim_threshold = 0.2
    fsim_threshold = 0.2

    # Check if the image similarity metrics pass the thresholds
    if results['rmse'] <= rmse_threshold and results['ssim'] >= ssim_threshold and results['fsim'] >= fsim_threshold:
        print("Image similarity test passed!")
        print("-------------------------------------------------------------------")
        return True
    else:
        print("Image similarity test failed.")
        print(f"RMSE: {results['rmse']:.4f} (threshold: {rmse_threshold})")
        print(f"SSIM: {results['ssim']:.4f} (threshold: {ssim_threshold})")
        print(f"FSIM: {results['fsim']:.4f} (threshold: {fsim_threshold})")
        return False
    print("-------------------------------------------------------------------")

if(background_image_similarity_check()):
    # Commenting out to allow multiple runs easier
    # Below code essentially identifies whether which image (with/without the background) is most similar to the prompt
    # In this use case, it is the test1.jpg. However, it can be the other one, in which the input code can be used / replaced with a programmatic automation (with more time).
    # FILE_PATH = input("Run src/clip_benchmarking/clip_benchmark.py for both the image and image without the background. Please enter the filepath that has the highest score:")
    FILE_PATH = "test1.jpg"

client = Anthropic(
    api_key="",
)

f = open('response.json') 

# returns JSON object as a list 
data = json.load(f)

multimodal_prompt = "Only respond in JSON. Here is a prompt: " + prompt +". For the attached image, please rank the following categories (on a 1-10 scale) in JSON format based on how similar the category in the image is to what the prompt intended: "

object_specified = data['objects']['specified']
if(object_specified):
    object_description = data['objects']['description']
    multimodal_prompt = multimodal_prompt + "objects, "
color_specified = data['colors']['specified']
if(color_specified):
    multimodal_prompt = multimodal_prompt + "colors, "
emotions_specified = data['emotions']['specified']
if(emotions_specified):
    multimodal_prompt = multimodal_prompt + "emotions, "
spatial_relations_specified = data['spatialRelations']['specified']
if(spatial_relations_specified):
    multimodal_prompt = multimodal_prompt + "spatial relations, "
activities_specified = data['activities']['specified']
if(activities_specified):
    multimodal_prompt = multimodal_prompt + "activities."

# I CHOSE NOT TO IMPLEMENT A MULTIMODAL CLAUDE MODEL REPEATEDLY IN THE CODE FOR THE COST BUT SEE AN IMPLEMENTATION HERE: https://docs.anthropic.com/en/docs/build-with-claude/vision
# Instead, I parsed it through with a model once and stored the response here:

MULTIMODAL_RESPONSE = {
    "objects": 9,
    "emotions": 8,
    "spatial_relations": 7,
    "activities": 9,
    "object_explanation": "The image contains a cat, oranges, and apples, which match the objects described in the prompt.",
    "emotion_explanation": "The cat in the image appears happy and excited, which closely matches the 'happy cat' described in the prompt.",
    "spatial_relation_explanation": "The image contains trees in the background, but the prompt specifies 'two trees behind it', which is not an exact match.",
    "activity_explanation": "The cat in the image is eating the oranges and apples, which directly matches the 'eating an orange and an apple' described in the prompt."
}

def object_detection(obj_description):

    # NOT USING MULTIPLE API CALLS DUE TO PRICING LIMITATIONS. SEE CODE BELOW. I PICKLED THE END RESULT.

#     model_url = (
#     "https://clarifai.com/clarifai/main/models/general-image-recognition"
#     )
#     model_prediction = Model(url=model_url, pat="").predict_by_filepath(
#         os.path.join("static", FILE_PATH), input_type="image"
#     )

#     # Get the output
#     parsed_data = [
#     {
#         "id": item.id,
#         "name": item.name,
#         "value": item.value,
#         "app_id": item.app_id
#     }
#     for item in model_prediction.outputs[0].data.concepts
# ]

#     print(parsed_data)

#     with open('data.pkl', 'wb') as file:
#         pickle.dump(parsed_data, file)

    with open('data.pkl', 'rb') as f:
        parsed_data = pickle.load(f)
    
    names = [item['name'] for item in parsed_data]


    # NOT CALLING MULTIPLE TIMES DUE TO API CALLS, SEE BELOW

    # message = client.messages.create(
    #     max_tokens=150,
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": "Here is the first list: " + str(obj_description) + ". See how many items in this list are represented in the following list and which items are not (ONLY look at items in the first list and see if they exist in second list, not vice versa): " + str(names) +". Generate a short message describing this.",
    #         }
    #     ],
    #     model="claude-3-opus-20240229",
    # )
    # print(message.content)

    message = "Comparing the items from the first list ['cat', 'orange', 'apple'] with the second list, we find:\n\n- 'cat' is present in the second list.\n- 'orange' and 'apple' are not directly mentioned in the second list. However, the second list contains related terms such as 'fruit', 'juicy', 'food', 'citrus', and 'tangerine', which are associated with oranges and apples.\n\nIn summary, 1 out of 3 items from the first list is directly represented in the second list, while the remaining 2 items have related terms present."

    return message

def color_detection():
    extracted = main_color_extract(FILE_PATH, prompt)
    color_names = [d['color_name'] for d in translate_colors(extracted)]
    
    # NOT CALLING MULTIPLE TIMES DUE TO API CALLS, SEE BELOW

    # message = client.messages.create(
    #     max_tokens=150,
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": "Here is a prompt: " + prompt + ". Here is a list of colors found in the image: " + str(color_names) + ". Give me a similarity ranking between how much of the prompt's colors are represented in the image (1-10) and a short explanation why. Return ONLY in JSON format.",
    #         }
    #     ],
    #     model="claude-3-opus-20240229",
    # )
    # print(message.content)

    results = {
        "colorSimilarity": 3,
        "explanation": "The image colors only partially match the prompt. While orange is present, likely representing the orange the cat is eating, the other prominent colors like dark gray and light gray do not seem to align with the prompt's description of trees and an apple. Key colors like green and red are missing."
    }
    
    return results['colorSimilarity'], results['explanation']

def get_numerical_average(dictionary):
    numbers = [value for value in dictionary.values() if isinstance(value, (int, float))]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)

def calc_claude_similarity_score():
    return get_numerical_average(MULTIMODAL_RESPONSE)

CLAUDE_SIMILARITY_SCORE = calc_claude_similarity_score()/10

def clip_similarity_score():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    image = preprocess(Image.open(os.path.join("static", FILE_PATH))).unsqueeze(0).to(device)
    text = clip.tokenize(["a a happy cat eating an orange and an apple with two trees behind it", "any image"]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0][0]
        return probs

CLIP_SIMILARITY_SCORE = clip_similarity_score()

TOTAL_SIMILARITY_SCORE = (CLAUDE_SIMILARITY_SCORE+CLIP_SIMILARITY_SCORE)/2

print("Claude Similarity Score (Ambiguity Accounted): " + str(CLAUDE_SIMILARITY_SCORE))

print("CLIP Similarity Score (Ambiguity Not Accounted For): " + str(CLIP_SIMILARITY_SCORE))

print("Simple Similarity Score (Weighted Avg): " + str(TOTAL_SIMILARITY_SCORE))

print("Manual Color Detection Results: ")
print(color_detection())

print("Manual Object Detection Results: ")
print(object_detection(object_description))

print("Multimodal Evaluation Results: ")
print(MULTIMODAL_RESPONSE)