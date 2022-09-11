from train import train
from generate import generate


a = input()
if a == "1":
    generator = generate("model")
    generator.gen(1000, st="вот так")
elif a == "2":
    tr = train("texts/text2.txt", "model", time=30)
    tr.tr()
