from flask import Flask, render_template, request
import pandas as pd
import textdistance
import re
from collections import Counter

app = Flask(__name__)

# Load and process text corpus
words = []
with open('autocorrect book.txt', 'r', encoding='utf-8') as f:
    data = f.read().lower()
    words = re.findall(r'\w+', data)  # Fixed invalid escape sequence

# Build vocabulary and probabilities
V = set(words)
words_freq_dict = Counter(words)
Total = sum(words_freq_dict.values())

probs = {k: v / Total for k, v in words_freq_dict.items()}

# Pre-instantiate distance metric
jaccard = textdistance.Jaccard(qval=2)

@app.route('/')
def index():
    return render_template('index.html', suggestions=None)


@app.route('/suggest', methods=['GET', 'POST'])
def suggest():
    if request.method == 'POST':
        keyword = request.form['keyword'].lower()
        if keyword:
            similarities = [1 - jaccard.distance(v, keyword) for v in words_freq_dict.keys()]
            df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
            df.columns = ['Word', 'Prob']
            df['Similarity'] = similarities
            suggestions = df.sort_values(['Similarity', 'Prob'], ascending=False)[['Word', 'Similarity']]
            suggestions_list = suggestions.head(10).to_dict('records')
            return render_template('index.html', suggestions=suggestions_list)
        else:
            return render_template('index.html', suggestions=[])
    else:
        # Handle GET request by just showing the empty form
        return render_template('index.html', suggestions=None)
    
if __name__ == '__main__':
    app.run(debug=True)




