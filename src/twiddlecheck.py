target = []
origin = []

target = input()
target = list(map(int, target.split()))
origin = input()
origin = list(map(int, origin.split()))


mod = 469762049
omega = 338628632

for j in range(256):
    for i in range(256):
        if (origin[j] * omega**i)%mod == target[j]:
            print(i) 