def evens_from_list(input_list):
    even_list = []
    for element in input_list:
        if element % 2 == 0:
            even_list.append(element)
    return even_list

def every_element_is_odd(input_list):
    for element in input_list:
        if element % 2 == 0:
            return False
    return True

def kth_largest_in_list(input_list, kth_largest):
    input_list.sort(reverse=True)
    return input_list[kth_largest - 1]

def cumavg_list(input_list):
    cumsum = 0
    cumavg = []
    for i, x in enumerate(input_list, 1):
        cumsum += x
        cumavg.append(cumsum / i)
    return cumavg

def element_wise_multiplication(input_list1, input_list2):
    if len(input_list1) != len(input_list2):
        raise ValueError("A listák hosszának egyformának kell lennie")
    output_list = []
    for i in range(len(input_list1)):
        product = input_list1[i] * input_list2[i]
        output_list.append(product)
    return output_list

def merge_lists(*lists):
    output_list = []
    for list in lists:
        output_list.extend(list)
    return output_list

def squared_odds(input_list):
    output_list = []
    for element in input_list:
        if element % 2 == 1:
            square = element ** 2
            output_list.append(square)
    return output_list

def reverse_sort_by_key(input_dict):
  sorted_dict = {}
  keys = list(input_dict.keys())
  values = list(input_dict.values())
  keys.sort(reverse=True)
  values.sort(reverse=True)
  for i in range(len(keys)):
    sorted_dict[keys[i]] = values[i]
  return sorted_dict

def sort_list_by_divisibility(input_list):
    output_dict = {}
    by_two = []
    by_five = []
    by_two_and_five = []
    by_none = []
    for element in input_list:
        if isinstance(element, int) and element > 0:
            if element % 2 == 0 and element % 5 == 0:
                by_two_and_five.append(element)
            elif element % 2 == 0:
                by_two.append(element)
            elif element % 5 == 0:
                by_five.append(element)
            else:
                by_none.append(element)
        else:
            raise ValueError("A lista csak pozitív egész számokat tartalmazhat")
    output_dict["by_two"] = by_two
    output_dict["by_five"] = by_five
    output_dict["by_two_and_five"] = by_two_and_five
    output_dict["by_none"] = by_none
    return output_dict