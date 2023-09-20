
def contains_value(input_list, element):
    for item in input_list:
        if item == element:
            return True
    return False


def number_of_elements_in_list(input_list):
    return len(input_list)


def remove_every_element_from_list(input_list):
    input_list.clear()


def reverse_list(input_list):
    reversed_list = []
    for i in range(len(input_list) - 1, -1, -1):
        reversed_list.append(input_list[i])
    return reversed_list



def odds_from_list(input_list):
    output_list=[]
    for element in input_list:
        if element % 2 == 1:
            output_list.append(element)
    return output_list



def number_of_odds_in_list(input_list):
    count=0
    for element in input_list:
        if element % 2 == 1:
            count += 1
    return count



def contains_odd(input_list):
    for element in input_list:
        if element % 2 == 1:
            return True
    return False


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


def sum_of_elements_in_list(input_list):
    total = 0.0
    for element in input_list:
        total += element
    return total


def cumsum_list(input_list):
    output_list = []
    current_sum = 0.0
    for element in input_list:
        current_sum += element
        output_list.append(current_sum)
    return output_list


def element_wise_sum(input_list1, input_list2):
    output_list = []
    min_length = min(len(input_list1), len(input_list2))
    for i in range(min_length):
        sum = input_list1[i] + input_list2[i]
        output_list.append(sum)
    return output_list


def subset_of_list(input_list, start_index, end_index):
    output_list = []
    for i in range(start_index, end_index + 1):
        output_list.append(input_list[i])
    return output_list


def every_nth(input_list, step_size):
    output_list = []
    for i in range(0, len(input_list), step_size):
        output_list.append(input_list[i])
    return output_list


def only_unique_in_list(input_list):
    unique_set = set()
    for element in input_list:
        if element in unique_set:
            return False
        unique_set.add(element)
    return True


def keep_unique(input_list):
    output_list = []
    for element in input_list:
        if element not in output_list:
            output_list.append(element)
    return output_list




def swap(input_list, first_index, second_index):
    temp = input_list[first_index]
    input_list[first_index] = input_list[second_index]
    input_list[second_index] = temp
    return input_list


def remove_element_by_value(input_list, value_to_remove):
    output_list = []
    for element in input_list:
        if element != value_to_remove:
            output_list.append(element)
    return output_list


def remove_element_by_index(input_list, index):
    output_list = []
    for i in range(len(input_list)):
        if i != index:
            output_list.append(input_list[i])
    return output_list


def multiply_every_element(input_list, multiplier):
    output_list = []
    for element in input_list:
        product = element * multiplier
        output_list.append(product)
    return output_list


