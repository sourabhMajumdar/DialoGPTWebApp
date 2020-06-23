from flask import Flask, render_template, request
from models import GPTBot

app = Flask(__name__)

bot = GPTBot()
convhistory = []
@app.route('/',methods=['GET','POST'])
def index() :
    global convhistory
    if request.method == 'GET' :
        convhistory = [('','')]
        pass

    if request.method == "POST" :
        text = request.form.get("input")

        # generate bot_response
        if text == 'reset' or text == 'quit' or text == 'q' :
            convhistory = [('','')]
            bot.purge_history()
        else :
            response = bot.generateResponse(text)
            convhistory.append(('USER',text))
            convhistory.append(('BOT',response))
        
    return render_template('index.html',convhistory=convhistory)

if __name__ == "__main__" :
    app.run(debug=True)