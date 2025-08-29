from google import genai
from PIL import Image
import io
import os, json
from google.genai import types

gemini_api_key = os.getenv('GEMINI_API_KEY')

print(gemini_api_key)

gemini_client = genai.Client(api_key=gemini_api_key)

def get_llm_response(image_data: bytes) -> str:
    image = Image.open(io.BytesIO(image_data))
    # implement the call to the Gemini API here
    # docs: https://ai.google.dev/gemini-api/docs/text-generation
    config = types.GenerateContentConfig(
    response_mime_type="application/json"
    )
    prompt = "Detect the all of the prominent items in the image. The box_2d should be [ymin, xmin, ymax, xmax] normalized to 0-1000."
    response = gemini_client.models.generate_content(model="gemini-2.5-flash",
                                            contents=[image, prompt],
                                            config=config
                                            )

    width, height = image.size
    bounding_boxes = json.loads(response.text)

    converted_bounding_boxes = []
    for bounding_box in bounding_boxes:
        abs_y1 = int(bounding_box["box_2d"][0]/1000 * height)
        abs_x1 = int(bounding_box["box_2d"][1]/1000 * width)
        abs_y2 = int(bounding_box["box_2d"][2]/1000 * height)
        abs_x2 = int(bounding_box["box_2d"][3]/1000 * width)
        converted_bounding_boxes.append([abs_x1, abs_y1, abs_x2, abs_y2])

    return len(converted_bounding_boxes) # return the nummber of objects 

    
