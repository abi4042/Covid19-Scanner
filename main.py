from flask import Flask, render_template, request
import pickle


model_file = open("model.pkl", "rb")
model = pickle.load(model_file)
model_file.close()


app = Flask(__name__)

@app.route('/', methods=["GET"])
def form():

    return render_template('index.html')


@app.route('/form', methods=["POST"])
def calculate_prob():

    if request.method == "POST":
        data = request.form
        print(data.keys)
        fever = int(data.get("fever"))
        tiredness = int(data.get("tiredness"))
        dry_cough = int(data.get("dry_cough"))
        shortness_of_breath = int(data.get("s_o_b"))
        aches_pain = int(data.get("a_p"))
        nasal_congestion = int(data.get("n_c"))
        runny_nose = int(data.get("runny_nose"))
        sore_throat = int(data.get("sore_throat"))
        diahrea = int(data.get("diahrea"))
        age = int( data.get("age"))

        input_data = [[fever, tiredness, dry_cough, aches_pain, nasal_congestion, runny_nose, sore_throat, diahrea, shortness_of_breath, age]]
        prob =  model.predict_proba(input_data)[0][1]
        print(prob)
        return render_template('display_prob.html',prob = prob)


if __name__ == "__main__":
    app.run(debug=True)
