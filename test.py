import MeCab

tokenizer = MeCab.Tagger()
a = tokenizer.parseToNode("아버지가 방에 들어가신다")
tokens = []
while a:
    if a.surface is not "":
        tokens.append(a.surface)
    a = a.next
print(tokens)