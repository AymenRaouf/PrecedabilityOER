# PreSAGE : Precedability Prediction Between Educational Resources

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

PreSAGE is a two- step method to identify precedability relations between educational resources. Our method structures the educational resources in an enriched Knowledge Graph (KG). Then, it uses a Graph Neural Network (GNN) model to predict precedability relations.

<b>This is the code for the paper recedability Prediction Between Educational Resources</b>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Installation

Download the project then use the package manager [pip](https://pip.pypa.io/en/stable/) to install the dependencies. Or you can simply use the <b>start.sh</b> file for linux users to create a virtual environment and install the packages.

```bash
pip install -r requirements.txt
```

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Data
- Due to the large size of the data folder, we were not able to upload it to Git. It will be uploaded to the organization's website after the paper review is completed. 

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Usage
- You can run the <b>Modeling/Ablation.ipynb</b> to test the PreSAGE model as well as its derived models for all datasets (except for Khan use <b>Modeling/Ablation_bis.ipynb</b>).
- The code for running the BERT and FastText baseline is found in the <b>Baselines</b> folder.
- The <b>Preprocessing</b> folder contains the code that was used to prepare the data.
- The <b>Output</b> folder contains the experimental results.
- The results can be seen and visualized using the files found on the <b>Evaluation</b> folder. 

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## License

[MIT](https://choosealicense.com/licenses/mit/)