f = "t.txt"

min = 9999
with open(f, "r") as f :
    for l in f.readlines() :
        if min > float(l) :
            min = float(l)
            
print(min)