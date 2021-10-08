import pandas as pd
a = ['adfasdf','asdfasdf']
df = pd.DataFrame(a, columns=['content'])
print(df)
print(type(df))
b= df['content']
print(type(b))