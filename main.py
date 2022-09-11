from train import train
from generate import generate


a = input()
if a == "generate":
    generator = generate("model")
    generator.gen(1000, st="вот так")
elif a == "train":
    tr = train("texts/text2.txt", "model", time=3)
    tr.tr()
