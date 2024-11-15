import csv
import matplotlib.pyplot as plt

# know the file path
filepath_01 = 'image_auto_encoder_01_log.csv'
filepath_001 = 'image_auto_encoder_001_log.csv'
filepath_10 = 'image_auto_encoder_10_log.csv'

filepath = filepath_10
lr = "0.10"
# use a context manager to open the file as a read-only
with open(filepath,"r") as file:
    # use csv library to read the csv and store it in a variable
    csv_reader = csv.reader(file)
    # read the first row of the csv if it contains headers and store those in a variable
    headers = next(csv_reader)
    print(f"Headers: {headers}")

    # A csv is best treated as a dictionary, so create an empty dict to hold the data
    data = {}
    # For every header in headers, set the header as a key and the value as an empty array.
    for header in headers:
        data[header] = []

    print(f"data: {data}")

    # Now for row in the file, for each header in header, append that data in the header's column to the empty array for the dicitonary 
    for row in csv_reader:
        for i, header in enumerate(headers):
            data[header].append(row[i])

print(f"Log Data: {data}")

# convert numerical values to a float for better plotting.
for i in data:
    data[i] = [float(element) for element in data[i]]
print(f"Log Data: {data}")


# Plot training & validation loss values
x = [epoch+1 for epoch in data['epoch']]
plt.subplot(2,1,1)
plt.plot(x,data['loss'],label="Loss")
plt.plot(x,data['val_loss'],label="Validation Loss")
plt.title(f'Model Loss vs. Epoch (lr: {lr})')
plt.ylabel('Loss')
# plt.xlabel('Epoch')
plt.legend()
# plt.show()

plt.subplot(2,1,2)
plt.plot(x,data['accuracy'],label="Accuracy")
plt.plot(x,data['val_accuracy'],label="Validation Accuracy")
plt.title(f'Model Accuracy vs. Epoch  (lr: {lr})')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

