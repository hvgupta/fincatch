from Extractor import getURLContent
import json

output = getURLContent("FinCatch_Sources_Medium.csv")

with open("output.json" , "w") as f:
    f.write(json.dumps(output))

