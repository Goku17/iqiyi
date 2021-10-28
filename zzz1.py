# cont_char_map_train = {}  # 一个样本中的所有角色
# for cont, char in zip(train_df["content"], train_df["character"]):
#     if cont not in cont_char_map_train:
#         cont_char_map_train[cont] = []
#     if isinstance(char, str):
#         cont_char_map_train[cont].append(char)
#
# cont_char_map_test = {}
# for cont, char in zip(test_df["content"], test_df["character"]):
#     if cont not in cont_char_map_test:
#         cont_char_map_test[cont] = []
#     if isinstance(char, str):
#         cont_char_map_test[cont].append(char)

'''
cp=accu, model=accu: 1.679068; 0.67835057885   ***tmp_optimum***
cp=accu, model=nonaccu: 1.667455; 0.67640259760
cp=nonaccu, model=accu:
cp=nonaccu, model=nonaccu: 1.666017; 0.67607338604
'''
