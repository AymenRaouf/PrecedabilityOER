# PreSAGE : Precedability Prediction Between Open Educational Resources

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

PreSAGE is a two- step method to identify precedability relations between open educational resources. Our method structures the educational resources in an enriched Knowledge Graph (KG). Then, it uses a Graph Neural Network (GNN) model to predict precedability relations.

<b>This is the code for the paper "Precedability Prediction Between Open Educational Resources"</b>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Installation

Download the project then use the package manager [pip](https://pip.pypa.io/en/stable/) to install the dependencies. Or you can simply use the <b>start.sh</b> file for linux users to create a virtual environment and install the packages.

```bash
pip install -r requirements.txt
```
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Data
- The data used is found in the <b>Data</b> folder.
- The type of data available :
    - Raw text data in the <b>Data/PUBLISHER/data</b>.
    - Precedence relations (Ground truth) <b>Data/PUBLISHER/precedence</b>
    - The initial node features (embeddings) are not included in this repository to keep its storage size minimal. However, the code to generate these embeddings is included.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Usage
- Generate initial node features using <b>Preprocessing/Features.ipynb</b>.
- You can run the <b>Results/Ablation.ipynb</b> to test the PreSAGE model as well as the other derived models for all datasets (except for Khan use <b>Modeling/Ablation_bis.ipynb</b>).
- The code for running the baselines BERT and FastText is found in the <b>Baselines</b> folder.
- The results can be seen and visualized using the files found in <b>Evaluation/Visualization.ipynb</b>. 
- The <b>Preprocessing</b> folder contains the code that was used to prepare the data.
- The <b>Output</b> folder contains the experimental results.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## License

[MIT](https://choosealicense.com/licenses/mit/)