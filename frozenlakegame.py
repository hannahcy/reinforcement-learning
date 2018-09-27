import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import time

"""frozenlakegame.py: Implementation of the frozen lake game for COSC470 Assignment 3.
"""


__author__      = "Lech Szymanski"
__license__ = "GPL"
__version__ = "1.0.0"
__email__ = "lechszym@cs.otago.ac.nz"

class frozenlakegame:

    # Constructor
    #
    # Input: R - the default reward for non-goal and non-water squares (cannot be -1 or 1), default: 0
    #        board_dims - dimensions of the board, default: 4x4
    #        max_num_hole - maximum number of holes that can appear on the board, default: 6
    def __init__(self, R=0, board_dims=[4,4], max_num_holes=6):

        self.__plot_handles = []
        self.H = board_dims[0]
        self.W = board_dims[1]

        self.__max_num_holes = max_num_holes

        self.num_actions = 4

        self.__actions_dirs = [[0,1],[1,0],[0,-1],[-1,0]]
        if np.abs(R) == 1:
            raise ValueError("The default state reward cannot be equal to +/- 1")

        self.__default_R = R
        self.__episode_count = 0
        self.__step_count = 0
        self.__wins_count = 0
        self.__win_ratio = []

        self.__fh = None
        self.__h = None
        self.__R = None
        self.__terminal = True

    # Reset envirnment - this will create a new board at random
    # place the player in the top-left corner and the goal in one of the
    # three remaining corners.  There water holes are placed at random such
    # that there is a path between the player and the goal
    def reset(self):

        if not self.__terminal:
            self.update_wins()

        # Randomly pick number of water holes - minimum is 1, maximum is 5
        num_holes = np.random.randint(3,self.__max_num_holes)

        # Randomly pick the location of the goal state (it's in one of three corners, left top one being where the
        # player starts
        goal_loc = np.random.randint(0,3)

        # Create a board layout
        while(True):
            R = np.ones((self.H,self.W))*self.__default_R

            if goal_loc == 0:
                goal_loc = (0,self.W-1)
            elif goal_loc == 1:
                goal_loc = (self.H-1,0)
            else:
                goal_loc = (self.H-1,self.W-1)

            hole_locs = np.random.permutation(self.H*self.W-1)+1
            hole_locs = hole_locs[:num_holes]

            for i in range(num_holes):
                x = hole_locs[i]%self.W
                y = int(np.floor(hole_locs[i]/self.W))

                R[y,x] = -1

            y,x = goal_loc

            # Check if a hole doesn't coincide with the goal square
            if R[y,x] == -1:
                continue

            R[y,x] = 1

            #Check if there is a path from the starting state to the goal
            reachable_states = [(0,0)]
            checked_states = list()

            goal_reachable = False
            while(not goal_reachable and reachable_states):

                state = reachable_states.pop(-1)
                checked_states.append(state)

                y,x = state
                for a_y,a_x in self.__actions_dirs:
                    y_sp = y+a_y
                    x_sp = x+a_x

                    if x_sp < 0 or x_sp >= self.W or y_sp < 0 or y_sp >= self.H:
                        continue

                    if R[y_sp,x_sp] == -1:
                        continue
                    elif R[y_sp,x_sp] == 1:
                        goal_reachable = True
                        break

                    new_state = (y_sp, x_sp)

                    if not new_state in checked_states:
                        reachable_states.append(new_state)

            if goal_reachable:
                break

        # Distribute rewards
        self.__R = R
        self.__state = np.zeros((self.W,self.H,3))
        for x in range(self.W):
            for y in range(self.H):
                self.set_state_pixel(y,x)

        self.__agent_loc = (0,0)
        self.set_agent_pixel()
        self.__episode_count += 1
        self.__step_count = 0
        self.__terminal = False

        return self.__state

    def set_state_pixel(self, y, x):
        self.__state[y,x,:]=0
        if self.__R[y, x] == 1:
            self.__state[y, x, 0] = 1.0
        elif self.__R[y, x] == -1:
            self.__state[y, x, 2] = 1.0
        else:
            self.__state[y, x, 1] = 1.0

    def set_agent_pixel(self):
        y,x = self.__agent_loc
        self.__state[y,x,:] = 1.0

    def update_wins(self):
        y, x = self.__agent_loc
        if self.__R[y, x] == 1:
            self.__wins_count += 1
        self.__win_ratio.append(np.sum(self.__wins_count) / self.__episode_count)

    # Checks if the environment is in terminal state
    #
    # Returns: bool - True if it's in terminal state, False otherwise
    def terminal(self):
        if self.__episode_count == 0:
            raise Exception('Need to reset the environment before use.')

        y,x = self.__agent_loc

        if np.abs(self.__R[y,x]) == 1:
            if not self.__terminal:
                self.update_wins()
                self.__terminal = True

            return True
        else:
            return False

    # Perform an action in the environment
    #
    # Input: a - an action index from 0 to num_actions-1
    #
    # Returns: np.array - a HxWx3 array corresponding to a HxW colour image
    #                     where white pixel indicates the player position, red pixel
    #                     the goal, blue pixels water holes and green pixels the ice.
    def step(self, a):
        if self.__episode_count == 0:
            raise Exception('Need to reset the environment before use.')

        if a<0 or a>3:
            raise Exception('Given action a=%d is out of bounds!  Valid actions are {0,...,%d}.' % (a, self.num_actions-1))

        r = np.random.rand()
        if a==0 or a==2:
            if r < 0.1:
                a = 1
            elif r < 0.2:
                a = 3
        elif a==1 or a==3:
            if r < 0.1:
                a = 0
            elif r < 0.2:
                a = 2

        y,x = self.__agent_loc

        if self.__terminal:
            return  self.__state, self.__R[y,x]


        self.set_state_pixel(y,x)
        if a==0:
            #Going north
            if y>0:
                y -= 1
        elif a==1:
            #Going east
            if x<self.W-1:
                x += 1
        elif a==2:
            #Going south
            if y<self.H-1:
                y += 1
        elif a==3:
            #Going west
            if x > 0:
                x -= 1

        self.__agent_loc = (y,x)
        self.set_agent_pixel()
        self.__step_count += 1
        if self.__step_count >= 100:
            self.__terminal = True

        return np.array(self.__state), np.array(self.__R[y,x])

    # Shows a visualisation of the game
    def show(self, s=None, blocking=False):
        if self.__episode_count == 0:
            raise Exception('Need to reset the environment before use.')

        if not self.__fh:
            self.__fh = plt.figure(figsize=(8, 4), dpi=100)

        if not self.__h:
            if s is not None:
                self.__h = [None, None]
                self.__ph = [[],[]]
            else:
                self.__h = [None]
                self.__ph = [[]]
        elif len(self.__h)==1 and s is not None:
            if self.h[0]:
                self.h[0].clear()
            self.__h = [None, None]
            self.__ph = [[], []]
        elif len(self.__h)==2 and s is None:
            if self.__h[0]:
                self.__fh.delaxes(self.__h[0])

            if self.__h[1]:
                self.__fh.delaxes(self.__h[1])

            self.__h = [None]
            self.__ph = [[]]

        if not self.__h[0]:
            if s is not None:
                self.__h[0]=self.__fh.add_subplot(1,2,2)
            else:
                self.__h[0]=self.__fh.add_subplot(1,1,1)
            self.__h[0].set_xlabel('Episode')
            self.__h[0].set_ylabel('Win ratio')
            self.__h[0].xaxis.set_major_locator(MaxNLocator(integer=True))

        if s is not None and not self.__h[1]:
            self.__h[1] = self.__fh.add_subplot(1, 2, 1)

            for x in range(self.W+1):
                self.__h[1].plot([x, x],[0, self.H],'k')

            for y in range(self.H+1):
                self.__h[1].plot([0, self.W],[y, y],'k')

            self.__h[1].set_axis_off()

        for p in self.__ph[0]:
            p.remove()
        self.__ph[0] = []

        N = len(self.__win_ratio)
        if N>1:
            x = np.linspace(1,N,N)
            h = self.__h[0].plot(x, self.__win_ratio,'b')
            self.__ph[0].append(h[0])
            self.__h[0].set_ylim(-0.01, 1.0)

        if s is not None:
            for p in self.__ph[1]:
                p.remove()

            self.__ph[1] = []

            for y in range(self.H):
                for x in range(self.W):
                    color = s[y,x]
                    rect = Rectangle((x, self.H - y - 1), 1, 1, angle=0.0, color=color)
                    self.__ph[1].append(self.__h[1].add_patch(rect))
            self.__h[1].set_title('Ep %d, Step %d' % (self.__episode_count, self.__step_count))

        if not blocking:
            plt.ion()
            plt.pause(0.01)
            time.sleep(0.01)
        else:
            plt.ioff()

        plt.show()
