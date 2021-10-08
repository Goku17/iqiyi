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