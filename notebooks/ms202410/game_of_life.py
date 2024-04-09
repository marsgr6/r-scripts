import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import bernoulli
import matplotlib.animation as animation

def main(args):
    """
    Uso: python automata.py <n> <init_prob>
    n: grid size (0, 0) to (n, n)
    init_prob: autoamta initialization probability 
    Ejemplo: python game_of_life.py 100 0.5
    """
    if len(args) != 2:
        print(main.__doc__)
    else:
        grid_size = int(args[0])

        prob_alive = float(args[1])

        G = nx.grid_2d_graph(grid_size,grid_size)

        pos = {(x,y): (y, -x) for x,y in G.nodes()}

        G.add_edges_from(
            [((x,y),(x+1, y+1)) for x in range(grid_size-1) for y in range(grid_size-1)] + 
            [((x+1,y),(x, y+1)) for x in range(grid_size-1) for y in range(grid_size-1)]
        )

        states = bernoulli.rvs(prob_alive, size=(grid_size, grid_size))
        #states = np.zeros((grid_size,grid_size), dtype='int')

        # Oscillator
        #x = np.array([5,5,5])
        #y = np.array([4,5,6])
        #states[(x,y)] = 1

        # slider
        """
        x = np.array([5,5])
        y = np.array([4,5])
        states[(x,y)] = 1
        x = np.array([4,4])
        y = np.array([5,6])
        states[(x,y)] = 1
        states[6,6] = 1
        """

        #plt.matshow(states)  # states in time 0
        #plt.pause(1)
        #mean_activity = [states.mean()]


        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)

        def anima(i):
            # modify axis
            ax1.clear()
            ax1.matshow(states, cmap='gray')
            # Update the automata for t steps
            states_t = states.copy()  # state in t-1
            for node in G:
                neighbors = np.array(list(G.neighbors(node)))
                activity = states_t[(neighbors[:,0], neighbors[:,1])].sum()
                # Cell is alive
                if states_t[node]:
                    # Isolaton and die
                    if activity < 2:
                        states[node] = 0
                    if activity >=4:
                        states[node] = 0
                # Cell is dead
                else: 
                    # Get born
                    if activity == 3:
                        states[node] = 1
            #mean_activity += [states.mean()]
            

        #where to animate, what to animate, how often to update
        ani = animation.FuncAnimation(fig, anima, interval = 10)
        plt.show()

        #plt.plot(mean_activity)


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
