import pandas as pd
df = pd.read_csv("org-data/training_data.csv")
df = df.fillna("")

text_col = []
for _, row in df.iterrows():
    prompt = str(row["text"])
    instruction = str(row["instruction"])
    input_query= str(row["input"])
    response = str(row["output"])

    if len(input_query.strip()) == 0:
        text = prompt + "### Instruction:\n" + instruction + "\n### Response:\n " + response
    else:
        text = prompt + "### Instruction:\n" + instruction + "\n### Input:\n" + input_query

    text_col.append(text)

df.loc[:,"text"] = text_col
print(df.head())

df.to_csv("train.csv",index=False)