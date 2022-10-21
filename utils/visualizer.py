import plotly.express as px
import pandas as pd

class Visualizer:
    def __init__(self):
        pass

    def visualize_words_in_topic(self, topic):
        
        topics_df = pd.DataFrame({"words": topic.words, "scores": topic.word_scores})
        fig = px.bar(topics_df, x="scores", y="words")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        fig.show()