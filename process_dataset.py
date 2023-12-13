import pandas
from tqdm import tqdm

data = pandas.read_csv('./dataset/RAW_recipes.csv')
data = data[["name","steps"]]
data.steps=data.steps.str[2:-2].str.split("', '").tolist()

def process_one_line(input : str) -> str:
    output = input.replace("   ", " ")
    output = output.replace("  ", " ")
    output = output.replace(" , ", ", ")
    return output.lower()

with open('./dataset/processed_dataset.txt', 'w') as data_write:
    for i, index in tqdm(enumerate(data.index), desc = 'Writing in file'):
        if(index % 50 == 0):    
            data_write.write(process_one_line(str(data['name'][i])) + ':\n')
            for j in data['steps'][i]:
                data_write.write('-' + process_one_line(str(j)) + '\n')
            data_write.write('\n')

