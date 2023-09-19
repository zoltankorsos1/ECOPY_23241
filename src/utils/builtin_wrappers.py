
1.
list=[17,18,3.14,'a','alma']

2.
r=range(1,11)
l=list(r)
print(l)

3.
l[1]

4.
l[0]

5.
max(l)

6.
l.index(10)

7.
def contains_values(input_list, element):
    for item in input_list:
        if item == element:
            return True
    return False
#%%
print(contains_values([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 11))

8.
def number_of_elements_in_list(input_list):
    return len(input_list)
#%%
print(number_of_elements_in_list((1,2,3,4,5,6,7,8,9,10)))

9.
def remove_every_element_from_list(input_list):
    input_list.clear()
#%%
l=[1,2,3,4,5,6,7,8,9,10]
remove_every_element_from_list(l)
print(l)

10.
def reverse_list(input_list):
    reversed_list = []
    for i in range(len(input_list) - 1, -1, -1):
        reversed_list.append(input_list[i])
    return reversed_list
#%%
print(reverse_list([1,2,3,4,5,6,7,8,9,10]))

    11.
def odds_from_list(input_list):
    output_list=[]
    for element in input_list:
        if element % 2 == 1:
            output_list.append(element)
    return output_list
#%%
l=[1,2,3,4,5,6,7,8,9,10]
result=odds_from_list(l)
print(result)

    12.
def number_of_odds_in_list(input_list):
    count=0
    for element in input_list:
        if element % 2 == 1:
            count += 1
    return count
#%%
l=[1,2,3,4,5,6,7,8,9,10]
result=number_of_odds_in_list(l)
print(result)

13.
def contains_odd(input_list):
    for element in input_list:
        if element % 2 == 1:
            return True
    return False
#%%
l=[1,2,3,4,5,6,7,8,9,10]
a=contains_odd(l)
print(a)

14.
def second_largest_in_list(input_list ):
    largest = max(input_list[0], input_list[1])
    second_largest = min(input_list[0], input_list[1])
    for i in range(2, len(input_list)):
        if input_list[i] > largest:
            second_largest = largest
            largest = input_list[i]
        elif input_list[i] > second_largest and input_list[i] < largest:
            second_largest = input_list[i]
    return second_largest
#%%
l=[1,2,3,4,5,6,7,8,9,10]
b=second_largest_in_list(l)
print(b)

15.
def sum_of_elements_in_list(input_list):
    total = 0.0
    for element in input_list:
        total += element
    return total
#%%
l=[1,2,3,4,5,6,7,8,9,10]
c=sum_of_elements_in_list(l)
print(c)

16.
def cumsum_list(input_list):
    output_list = []
    current_sum = 0.0
    for element in input_list:
        current_sum += element
        output_list.append(current_sum)
    return output_list
#%%
l=[1,2,3,4,5,6,7,8,9,10]
d=cumsum_list(l)
print(d)


17.

def element_wise_sum(input_list1, input_list2):
    output_list = []
    min_length = min(len(input_list1), len(input_list2))
    for i in range(min_length):
        sum = input_list1[i] + input_list2[i]
        output_list.append(sum)
    return output_list
input_list1 = [1, 2, 3,4,5,6,7,8,9,10]
input_list2 = [11,12,13,14,15,16,17,18,19,20]
f = element_wise_sum(input_list1, input_list2)
print(f)

18.
def subset_of_list(input_list, start_index, end_index):
    output_list = []
    for i in range(start_index, end_index + 1):
        output_list.append(input_list[i])
    return output_list
#%%
input_list = [1, 2, 3,4,5,6,7,8,9,10]
start_index = 1
end_index = 4
g = subset_of_list(input_list, start_index, end_index)
print(g)

19.
def every_nth(input_list, step_size):
    output_list = []
    for i in range(0, len(input_list), step_size):
        output_list.append(input_list[i])
    return output_list
#%%
l = [1, 2, 3, 4, 5,6,7,8,9,10]
step_size = 2
h = every_nth(l, step_size)
print(h)



20.
def only_unique_in_list(input_list):
    unique_set = set()
    for element in input_list:
        if element in unique_set:
            return False
        unique_set.add(element)
    return True
#%%
l = [1, 2, 3, 4, 5,6,7,8,9,10]
j = only_unique_in_list(l)
print(j)


21.

def keep_unique(input_list):
    output_list = []
    for element in input_list:
        if element not in output_list:
            output_list.append(element)
    return output_list
#%%
l = [1,1, 2, 3, 4, 5,6,7,8,9,10]
k = keep_unique(l)
print(k)

22.
def swap(input_list, first_index, second_index):
    temp = input_list[first_index]
    input_list[first_index] = input_list[second_index]
    input_list[second_index] = temp
    return input_list
#%%
l = [1, 2, 3, 4, 5,6,7,8,9,10]
first_index = 1
second_index = 4
y = swap(l, first_index, second_index)
print(y)

23.
def remove_element_by_value(input_list, value_to_remove):
    output_list = []
    for element in input_list:
        if element != value_to_remove:
            output_list.append(element)
    return output_list
#%%
l = [1, 2, 3, 4, 5,6,7,8,9,10]
value_to_remove=1
c=remove_element_by_value(l, value_to_remove)
print(c)

24.
def remove_element_by_index(input_list, index):
    output_list = []
    for i in range(len(input_list)):
        if i != index:
            output_list.append(input_list[i])
    return output_list
#%%
l = [1, 2, 3, 4, 5,6,7,8,9,10]
value_to_remove=1
index=0
v=remove_element_by_index(l, index)
print(v)


25.
def multiply_every_element(input_list, multiplier):
    output_list = []
    for element in input_list:
        product = element * multiplier
        output_list.append(product)
    return output_list
#%%
l = [1, 2, 3, 4, 5,6,7,8,9,10]
multiplier=10
b=multiply_every_element(l, multiplier)
print(b)


Dic

1.
