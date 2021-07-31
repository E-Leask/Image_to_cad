class Value:
    def __init__(self, v=None):
        self.v = v

class Vertex:
    def __init__(self, x=None,y=None,connections=None):
        self.x=x
        self.y=y
        self.c=connections
a=[[1],[2],[3],[4]]
b=a[0]
b[0]=5
print(a)
print(b)