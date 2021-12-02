from RL_brain1 import RL
from maze_env1 import Maze
import tensorflow as tf
import numpy as np
 
def run():
    step = 0
    for episode in range(10):
        observation = env.reset()
        #print(observation)
        while True:
            env.render()
            action = DQL.choose_action(observation)
            observation_, reward, done = env.step(action)
            DQL.store_transition(observation, action, reward, observation_)
            #print()

            if(step > 250) and (step % 5 ==0):
                DQL.learn()

            step += 1
            observation = observation_
            if done:
                break
    print("done")

if __name__ == '__main__':
    env = Maze()
    DQL = RL(env.n_actions, env.n_features)
    #x = np.array([[45.0,45.0]])
    #print(DQL.getNet(x))
    env.after(10, run())
    env.mainloop()
    #x = np.array([[45.0,45.0]])
    #print(DQL.getNet(x))
    #print(DQL.getNet([45 45]))