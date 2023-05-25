import tempfile
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import dotenv_values
from tools.analyze_script import postProcess_output, analyze_text
from tools.transcriber import transcript
from tools.audio import AudioAnalyzer, audioReport
from tools.emotion import EmotionAnalyzer
from tools.eye_engagement import EyeEngagementAnalyzer

config = dotenv_values('.env')

FER_MODEL = config.get('FER_MODEL', '')
GPT4J_MODEL = config.get('GPT4J_MODEL', '')
OPENAI_API_KEY = config.get('OPENAI_API_KEY', '')
SHAPE_PREDICTOR_MODEL=config.get('SHAPE_PREDICTOR_MODEL','')


def main():
    st.title("Interview Screener")
    file_path = st.file_uploader("Choose a file", type=['mp4', 'mkv'], disabled=False)

    if file_path is None:
        return

    *_, ext = file_path.name.split('.')

    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{ext}') as input_video:
        input_video.write(file_path.getvalue())
        video_path = input_video.name

    # st.write(f"Video File:\t{video_path}")

    del(file_path)
    
    text = transcript(video_path, OPENAI_API_KEY)

    st.title("Transcript:")
    st.write(text)

    output = analyze_text('text-davinci-003', OPENAI_API_KEY, text)
    
    st.title("Review")
    st.write(postProcess_output(output=output))

    audioAnalysis = AudioAnalyzer(video_path)
    st.title("Audio Analysis Report")

    data = {
            'Metrics': ['Silence Percentage', 'Average Confidence', 'Emphasis Percentage', 'Speech-to-Silence Ratio'],
            'Value': audioReport(audioAnalysis)
    }

    df = pd.DataFrame(data)
    df.set_index('Metrics', inplace=True)
    st.table(df.rename_axis(None))
    st.pyplot(audioAnalysis.plot_all_charts())

    del(data, df)

    st.title("Emotion Aanalysis Report")
    emotionAnalysis = EmotionAnalyzer(FER_MODEL, video_path)
    emotionAnalysis.analyze_video()
    emotions_percentage = emotionAnalysis.calculate_emotion_percentages()

    # Create a DataFrame from the dictionary
    df = pd.DataFrame.from_dict(emotions_percentage, orient='index', columns=['Percentage'])
    df.index.name = 'Emotion'

    # Display the DataFrame as a table
    st.dataframe(df)

    pos_emotion, neg_emotion = emotionAnalysis.calculate_positive_negative_percentages()
    st.write("Total Emotion Percentages:\n\n")
    st.write(f'Positive Emotions: {pos_emotion:.2f}%\n\n')
    st.write(f'Negative Emotions: {neg_emotion:.2f}%\n\n')

    st.pyplot(emotionAnalysis.plot())

    st.title('Eye Engagement Report')

    eyeAanalyser = EyeEngagementAnalyzer(video_path, SHAPE_PREDICTOR_MODEL, {
        'low': 0.2,
        'medium': 0.4
    })

    eyeAanalyser.process_video()
    eyeAanalyser.calculate_eye_engagement()

    df = eyeAanalyser.data_to_dataframe()
    overall_engagement = eyeAanalyser.calculate_engagement_level(df['EyeEngagement'].mean())
    st.write(f"Overall Engagement Level: {overall_engagement}")

    # Line plot
    fig, ax = plt.subplots()
    df.plot(x='Timestamp', y='EyeEngagement', kind='line', ax=ax)
    st.pyplot(fig)

    # Bar plot
    fig2, ax2 = plt.subplots()
    df['EngagementLevel'].value_counts().plot(kind='bar', ax=ax2)
    st.pyplot(fig2)

if __name__ == "__main__":
    main()