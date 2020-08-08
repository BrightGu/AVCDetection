



a_map={'q':[1,1,2],'w':[2,2,2]}
b_map={'z':[1,1,1],'x':[3,4,5]}
for label, map in zip(['a', 'b'],[a_map, b_map]):

    for label1, values in map.items():
        # high or low
        temp_list = [(value - 1) / 2 for value in values]
        map[label1] = temp_list
print(a_map)
print(b_map)
print("h")