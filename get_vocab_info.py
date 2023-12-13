with open("./dataset/processed_dataset.txt","r") as doc:
    output = doc.read()
    print(sorted(list((set(output)))))
    print(len(set(output)))