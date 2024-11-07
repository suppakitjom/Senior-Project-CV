from openai import OpenAI
from config import OPENAI_API_KEY, AZURE_SPEECH_KEY, AZURE_SPEECH_REGION
import azure.cognitiveservices.speech as speechsdk

client = OpenAI(api_key=OPENAI_API_KEY)

def generate_golf_feedback(feedback_data):
    # Format the feedback data into a text string
    feedback_text = ""
    for position, comments in feedback_data.items():
        feedback_text += f"{position}:\n" + "\n".join(comments) + "\n\n"
    
    # Create the request to OpenAI's API for summarization
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an encouraging golf coach giving simple feedback to a beginner. Make the feedback conversational, avoid using numbers, and keep it very easy to understand. Make sure to keep it very short and to the point."},
            {
                "role": "user",
                "content": f"Here’s some golf swing feedback. Summarize it in a way that's easy to understand:\n\n{feedback_text}"
            }
        ]
    )
    
    # Extract the summarized script from the response
    script = completion.choices[0].message.content
    print("Generated Script:\n", script)

    # Generate TTS audio
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    
    voices = [
        'en-US-Davis:DragonHDLatestNeural',
        'en-US-Andrew:DragonHDLatestNeural', # me likey
        'en-US-Andrew2:DragonHDLatestNeural',
        'en-US-Aria:DragonHDLatestNeural',
        'en-US-Ava:DragonHDLatestNeural',
        'en-US-Brian:DragonHDLatestNeural',
        'en-US-Steffan:DragonHDLatestNeural' # me likey
    ]
    speech_config.speech_synthesis_voice_name = voices[-1]

    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

    result = speech_synthesizer.speak_text_async(script).get()

if __name__ == "__main__":
    # Sample feedback data
    feedback_data = {
        'Address': [
            'Spine angle is 72.38 degrees. Try to keep a more upright spine.',
            'Foot width is wider than shoulder by 1.41 units. Good stance width for stability.',
            'Left arm angle is 165.53 degrees. Nice and straight!'
        ],
        'Toe-up': [
            'Left arm angle is 167.88 degrees. Nice and straight!',
            'Head is stable in the Toe-up position. Good job!'
        ],
        'Mid-backswing (arm parallel)': [
            'Left arm angle is 141.72 degrees. Nice and straight!',
            'Head is stable in the Mid-backswing position. Good job!'
        ],
        'Top': [
            'Left arm angle is 133.75 degrees. Try to keep your left arm straighter for better control.'
        ]
    }

    # Run the function
    generate_golf_feedback(feedback_data)