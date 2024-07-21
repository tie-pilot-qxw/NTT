import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def draw_data_flow(lgp):
    
    deg = 4
    n = 4096
    blockdim = 2**deg // 2
    color = list(mcolors.TABLEAU_COLORS.keys())
    for blockid in range(n//(blockdim * 2)):
        for threadid in range(blockdim):
            lid = threadid
            lsize = blockdim
            index = blockid
            t = n >> deg
            p = 1 << lgp
            k = index & (p - 1)

            count = 1 << deg
            counth = count >> 1

            counts = count // lsize * lid
            counte = counts + count // lsize

            x = index
            y = ((index - k) << deg) + k

            for i in range(counts, counte):
                plt.scatter(i*t + x, 1)
                # print(i*t + x, end = ' ')

            for i in range(counts // 2, counte // 2):
                plt.scatter(i*p + y, 2)
                plt.scatter((i + counth)*p + y, 2)
                print(i*p + y, p*(i+counth) + y, end=' ')
        print(" ")


    # plt.show()

# Call the function to draw the data flow diagram
draw_data_flow(4)