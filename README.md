1. Install required libraries and dependencies.

`pip install -r requirements.txt`

2. Download our model from SFU Vault: [distilbert-bigru-emotion.pt](https://vault.sfu.ca/index.php/s/XfnXASxEY5bc9H9).

3. Or build the model in your machine by running `distilbert_bigru.py`. 

    - The whole process will take for around 15 hours：there are 5 epochs in total, and the model (from the epoch with the best validation loss) will be saved in the `projects` directory.

    - You can alter the number of epoch you want in the hyperparameters section (from line 34 in `distilbert-bigru.py`), usually training for only 1 epoch can give you good result with 95.07% accuracy on the validation set. 

4. Dataset: the dataset `full.csv` is the dataset we used for training and validating the model. the dataset `test.txt` is the dataset we used for testing the mode performance. 

5. Please run the `check.py` to evaluate the model performance based on accuracy. A output file `output.txt` will be generated in the current directory `project`. Each line in `output.txt` is formatted as: `[original sentence];[predict label]`. For example: 
`im feeling rather rotten so im not very ambitious right now;sadness`. The reference file `reference.txt` is also provided. We will evaluate the predict accuracy based on the number of correct predict, by comparing the output file and the reference file. 

