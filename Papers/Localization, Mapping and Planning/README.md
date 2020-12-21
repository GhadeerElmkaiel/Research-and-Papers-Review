# Papers Review

## Rapid Exploration With Multi-Rotors

In this paper, the main idea was to adjust the Classic Frontier exploration approach by considering the direction of the robot's speed vector, and treating it as if it has an **inertia** *(the algorithm tends to plane the robot's trajectory in a way to ensure  that the change in the speed as minimum as possible)*. Even though this might lead to more distance traveled by the robot compared with classical Frontier exploration. The overall time is less, which leads to faster exploration.

## PRM-RL: Long-range Robotic Navigation Tasks by Combining Reinforcement Learning and Sampling-based Planning

In this paper, The main idea to achieve long distance planning and navigation tasks using a combination of Sampling methods and Reinforcement learning agent.
The Sampling method is used to provide the roadmaps which connect robot configurations that can be successfully navigated by the RL agent, while RL agent learns to short-range navigation policy. 

## CONSERVATIVE SAFETY CRITICS FOR EXPLORATION

In this paper, the authors present a model designed safe exploration for Reinforcement Learning. This is done by building a model that estimate the **safety** of a certain state. Safe exploration then is achieved by training the safety estimation model to overestimate the probability of failure.

## ACTIVE NEURAL LOCALIZATION

In this paper the Author *"the same author of (learning to explore using active neural slam)"* presents an improvement method for SLAM algorithms, making the algorithm **Active**, which means that the algorithm is **not only** responsible of localizing the robot, **but** also, it is responsible for deciding the next action which increase the probability of getting better localization. 
