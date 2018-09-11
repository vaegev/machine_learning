from A import A

class B(A):
    def __init__(self, name, surname):
        A.__init__(self, name)
        self.surname = surname


    def getFullName(self):
        return self.surname + '\n' + self.name

