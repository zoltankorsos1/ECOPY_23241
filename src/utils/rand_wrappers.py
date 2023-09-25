
def hundred_small_random():
    # létrehozunk egy üres listát
    output_list = []
    # 100-szor ismételjük a következőt
    for i in range(100):
        # generálunk egy véletlen számot 0 és 1 között
        x = random.uniform(0, 1)
        # hozzáadjuk a számot a listához
        output_list.append(x)
    # visszaadjuk a listát
    return output_list

random.seed(42) # beállítjuk a random seed értékét 42-re
n = random.randint(1, 100) # generálunk egy véletlen egész számot 1 és 100 között
print(n)