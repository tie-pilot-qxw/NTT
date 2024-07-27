target = []
origin = []

target = input()
target = list(map(int, target.split()))
origin = input()
origin = list(map(int, origin.split()))


mod = 469762049
omega = 426037461

for j in range(16):
    for i in range(16):
        if (origin[j] * omega**i)%mod == target[j]:
            print(i) 