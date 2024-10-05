import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from distilbert_bigru import DistilBERTBiGRU

OUTPUTFILE = "output.txt"
TESTFILE = "test.txt"
REFERENCEFILE = "reference.txt"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

# Load model
model = DistilBERTBiGRU(bert_model_name='distilbert-base-uncased', hidden_size=256, num_layers=4, num_classes=6)  
model.load_state_dict(torch.load("distilbert-bigru-emotion.pt")) 
model.eval()

# Load testing dataset
test_data = []

# Load testing data from file
with open(TESTFILE, "r") as test_file:
    for line in test_file:
        sentence = line.strip()
        test_data.append(sentence)

# Define mapping of labels to indices
label_to_index = {'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}

# Define mapping of indices to labels.
index_to_label = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']


# Define predict label function.
def predict(model, test_data, OUTPUTFILE):
    with open(OUTPUTFILE, 'w') as output_file:
        # Initialize tqdm progress bar
        progress_bar = tqdm(test_data, desc="Testing")
        
        with torch.no_grad():
            for sentence in progress_bar: 
                # Tokenize the sentence.
                tokenized_data = tokenizer(sentence, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

                # Predict output.
                ouputs = model(**tokenized_data)
                
                # Convert output to predicted label.
                predicted_index = torch.argmax(ouputs).item()
                
                # Convert the predicted label to text.
                predicted_label = index_to_label[predicted_index]  
                
                # Write original sentence and predicted label to output file
                output_file.write(f"{sentence};{predicted_label}\n")
                
                
# Define the function to compare the difference between reference.txt and output.txt => Calculate accuracy.
def evaluate(OUTPUTFILE, REFERENCEFILE):
    # Open both files.
    with open(OUTPUTFILE, 'r') as output, open(REFERENCEFILE, 'r') as ref:
        
        # Read the lines from each file.
        lines_output = output.readlines()
        lines_ref = ref.readlines()

        # Count the number of lines in each file.
        num_lines_output = len(lines_output)
        num_lines_ref = len(lines_ref)

        # Initialize counter for different lines.
        num_identical_lines = 0

        # Iterate through lines and compare
        min_num_lines = min(num_lines_output, num_lines_ref)
        for i in range(min_num_lines):
            if lines_output[i] == lines_ref[i]:
                num_identical_lines += 1
    
    # Return accuracy        
    return num_identical_lines / num_lines_ref
                
                
# Start testing
predict(model, test_data, OUTPUTFILE)
accuracy = evaluate(OUTPUTFILE, REFERENCEFILE)
print("Accuracy:", accuracy)