from flask import Flask, render_template, request
import History_of_Nepal  # Import your converted Python script

# app = Flask(__name__)
from flask import Flask
app = Flask(__name__)

# App configuration (optional)
app.config["DEBUG"] = True

@app.route('/')
def index():
    return render_template('index.html')  # Pass None initially

@app.route('/get_history', methods=['POST'])
def get_history():
    question = request.form['question']
    answer = History_of_Nepal.get_answer(question)  # Call the function to get the answer
    return render_template('index.html', answer=answer)  # Render the template with the answer

if __name__ == '__main__':
    app.run(debug=True)
