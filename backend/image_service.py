
import ollama


def analyze_image(
    image_file,
    model="llama3.2-vision"
):

    image_bytes = image_file.getvalue()

    response = ollama.chat(

        model=model,

        messages=[

            {
                "role": "user",

                "content": """
Analyze this agricultural image.

Identify:

1. What type of crop or plant may be visible.
2. Visible symptoms.
3. Possible causes.
4. Possible disease or pest if there is enough evidence.
5. Recommended next steps.

Do not claim a definite diagnosis.

If the image is unclear, say that the image quality is insufficient.

Focus only on agriculture.
""",

                "images": [
                    image_bytes
                ]
            }

        ]
    )

    return response["message"]["content"]



