import pandas as pd
import matplotlib.pyplot as plt
from nltk import word_tokenize

class FillerWordAnalyzer:
    def __init__(self, df):
        self.df = df
        self.df['text'] = self.df['text'].astype(str)
        self.filler_words = ["um", "uh", "like", "you know", "so", "basically", "actually", "literally", "well", "kinda",
                            "sort of", "I mean", "really", "totally", "seriously", "just", "pretty much", "perhaps", "maybe",
                            "possibly", "definitely", "absolutely", "obviously", "honestly", "okay", "alright", "oh", "ah",
                            "right", "wrong", "anyway", "anyhow", "well", "now", "then", "therefore", "thus", "however",
                            "nevertheless", "nonetheless", "besides", "moreover", "furthermore", "instead", "meanwhile",
                            "otherwise", "anyways", "anyhow", "anyhow", "consequently", "apparently", "unfortunately",
                            "fortunately", "surely", "certainly", "probably", "maybe", "actually", "simply", "truly",
                            "specifically", "honestly", "frankly", "anyhoo", "uh-huh", "huh", "uh-uh", "mmm-hmm"]

    def calculate_filler_word_frequency(self):
        self.df['tokens'] = self.df['text'].apply(word_tokenize)
        
        self.df['is_filler'] = self.df['tokens'].apply(lambda tokens: [word.lower() in self.filler_words for word in tokens])
        
        filler_count = self.df['is_filler'].apply(sum).sum()
        total_words = self.df['tokens'].apply(len).sum()
        filler_percentage = (filler_count / total_words) * 100
        
        return filler_count, filler_percentage
    
    def plot_filler_word_usage(self):
        self.df['start_time'] = pd.to_datetime(self.df['start'], unit='ms')
        self.df['end_time'] = pd.to_datetime(self.df['end'], unit='ms')
        
        self.df['filler_count'] = self.df['is_filler'].apply(sum)
        
        fig, ax = plt.subplots()
        ax.plot(self.df['start_time'], self.df['filler_count'])
        ax.set_xlabel('Time')
        ax.set_ylabel('Filler Word Count')
        ax.set_title('Filler Word Usage Over Time')
        ax.tick_params(rotation=45)
        plt.tight_layout()
        
        return fig

if __name__ == '__main__':
    # Pass the DataFrame to the FillerWordAnalyzer class
    df = pd.read_csv('transcription.csv')
    analyzer = FillerWordAnalyzer(df)

    # Calculate filler word frequency and percentage
    filler_count, filler_percentage = analyzer.calculate_filler_word_frequency()
    print(f"Filler word frequency: {filler_count}")
    print(f"Filler word percentage: {filler_percentage:.2f}%")

    # Plot the graph
    g = analyzer.plot_filler_word_usage()
    g.show()

    import time
    time.sleep(5)