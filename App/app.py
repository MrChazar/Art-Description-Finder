import pandas as pd
from flask import Flask, render_template, request
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import pandas
import create_embeddings as ce

app = Flask(__name__)
client = chromadb.PersistentClient(path="Model/Data/Chroma")

# Chroma instance initialization


# Endpoints initialization
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/tutorial')
def tutoria():
    return render_template('tutorial.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/submit', methods=['POST'])
def submit():

    prompt = request.form['prompt']

    collection = client.get_collection(name="embeddings", embedding_function=ce.get_embedding_function())

    print(collection)
    results = collection.query(
        query_texts=[prompt],  # Chroma will embed this for you
        n_results=1  # how many results to return
    )

    print(results)

    outcome = pd.read_csv('Model/Data/outcome.csv', sep=';')

    ids = results['ids']
    documents = results['documents']
    # for "unpacking" list
    ids = ''.join([str(item) for sublist in ids for item in sublist])
    documents = ''.join([str(item) for sublist in documents for item in sublist])
    ids = ids[:-2]
    print(f"ids {ids}")
    outcome_result = outcome.loc[outcome['game'] == ids]

    result  = []
    result.append(outcome[outcome['game'] == ids]['style'].values)
    result.append(outcome[outcome['game'] == ids]['style_description'].values)
    result.append(ids)
    result.append(documents)

    print(f"outcome{outcome_result}")

    return render_template('index.html', result=result)



if __name__ == '__main__':
    # build database
    ce.build_database()
    app.run(debug=True)
