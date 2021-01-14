class Parent:
    def __init__(self):
        self.a = 5
        print('b')
        self.display()
    def display(self):
        print('parent')

class Child(Parent):
    def __init__(self):
        self.a = 2
        super().__init__()
    def display(self):
        print('child')


Child()