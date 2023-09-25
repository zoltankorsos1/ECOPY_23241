class FirstClass:
    # definiálunk egy osztály attribútumot, ami minden példányra vonatkozik
    class_attribute = "This is a class attribute"
    def __init__(self, instance_attribute):
        # definiálunk egy példány attribútumot, ami csak az adott példányra vonatkozik
        self.instance_attribute = instance_attribute
    # definiálunk egy normál metódust, ami valamilyen műveletet végez az osztály példányain
    def instance_method(self):
        # kiírjuk az osztály és a példány attribútumokat
        print(self.class_attribute)
        print(self.instance_attribute)