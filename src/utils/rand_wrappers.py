
def hundred_small_random():
    import random
    # létrehozunk egy üres listát
    output_list = []
    # 100-szor ismételjük a következőt
    for i in range(100):
        x = random.uniform(0, 1)
        # hozzáadjuk a számot a listához
        output_list.append(x)
    # visszaadjuk a listát
    return output_list