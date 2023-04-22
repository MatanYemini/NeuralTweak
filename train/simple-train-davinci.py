import openai
import os
import re

# Set OpenAI API key
openai.api_key = os.environ["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"]

# Set training parameters
model_name = "text-davinci-002"
batch_size = 4
learning_rate = 5e-5
num_epochs = 3
max_seq_length = 512

# Load training data
with open("training_data.txt", "r") as f:
    text = f.read()

# Preprocess training data
text = re.sub(r"\n", " ", text)
text = re.sub(r"\s+", " ", text)

# Initialize model
model = openai.Completion.create(
    engine=model_name,
    prompt="",
    max_tokens=max_seq_length,
    n=1,
    stop=None,
    temperature=0.5,
)

# Train model
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}")
    for i in range(0, len(text), batch_size * max_seq_length):
        batch_text = text[i:i + batch_size * max_seq_length]
        response = openai.Completion.create(
            engine=model_name,
            prompt=batch_text,
            max_tokens=max_seq_length,
            n=1,
            stop=None,
            temperature=0.5,
            frequency_penalty=0,
            presence_penalty=0,
            learning_rate=learning_rate,
            include_prefix=False,
        )
        loss = response.choices[0].text
        print(f"Batch {i // (batch_size * max_seq_length) + 1} loss: {loss}")


def train_model(processed_files_dir):
    files = os.listdir(processed_files_dir)
    print("Found files to train the model on: ", files)
    files = [i for i in files if '.txt' in i]
    for file in files:
        try:
            print("Trying to traing the model on: ", file)
            with pdfplumber.open(src_dir+file) as pdf:
                output = ''
                for page in pdf.pages:
                    output += page.extract_text()
                    output += '\n\nNEW PAGE\n\n'  # change this for your page demarcation
                save_file(dest_dir+file.replace('.pdf','.txt'), output.strip())
                print("Processed file: ", file)
        except Exception as oops:
            print(oops, file)

if __name__ == '__main__':
    train_model('../assets/pre-processing/PDFs/', '../assets/post-processing')