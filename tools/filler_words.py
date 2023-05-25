from nltk import word_tokenize

class FillerWordAnalyzer:
    def __init__(self, text_list):
        self.text_list = text_list
        self.filler_words = ["um", "uh", "like", "you know", "so", "basically", "actually", "literally", "well", "kinda",
                            "sort of", "I mean", "really", "totally", "seriously", "just", "pretty much", "perhaps", "maybe",
                            "possibly", "definitely", "absolutely", "obviously", "honestly", "okay", "alright", "oh", "ah",
                            "right", "wrong", "anyway", "anyhow", "well", "now", "then", "therefore", "thus", "however",
                            "nevertheless", "nonetheless", "besides", "moreover", "furthermore", "instead", "meanwhile",
                            "otherwise", "anyways", "anyhow", "anyhow", "consequently", "apparently", "unfortunately",
                            "fortunately", "surely", "certainly", "probably", "maybe", "actually", "simply", "truly",
                            "specifically", "honestly", "frankly", "anyhoo", "uh-huh", "huh", "uh-uh", "mmm-hmm"]

    def calculate_filler_word_frequency(self):
        total_words = 0
        filler_count = 0

        for text in self.text_list:
            tokens = word_tokenize(text)
            total_words += len(tokens)
            filler_count += sum(word.lower() in self.filler_words for word in tokens)

        filler_percentage = (filler_count / total_words) * 100
        
        return filler_count, filler_percentage

if __name__ == '__main__':
    # Pass the list of texts to the FillerWordAnalyzer class
    text_list = [
        "Hi, there woah",
        "uhm, woah",
        "wow, it's great",
        # Add more texts here
    ]
    analyzer = FillerWordAnalyzer(text_list)

    # Calculate filler word frequency and percentage
    filler_count, filler_percentage = analyzer.calculate_filler_word_frequency()
    print(f"Filler word frequency: {filler_count}")
    print(f"Filler word percentage: {filler_percentage:.2f}%")