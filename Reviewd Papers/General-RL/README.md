# General-RL

## Reinforcement Learning using Guided Observability  

This paper mainly discusses the problem of Training RL agents for partially observable tasks.  
The authors showed that other methods deal with partial observations in different ways, some focus on improving the memory representation to compensate for the loss in observation, while others assume a strong knowledge about the type of partial observability which allows for creating dedicated models for each case. Other naive methods consider the Partial Observation (PO) as a Full Observation (FO) which leads to suboptimal policies.  
In This work, they approach this problems by using a mix of FOs and POs during the training step, and gradually moving from full FOs to full POs by linearly replace a FO with a corresponding PO (zeros for the unobservable states, and additional flag to represent that this is a PO vector).  

### Questions

1- when changing from FO to PO, why change the full vector a tonce (replace all unobservable values with zeros)? In the case of multiple unobservable values why not to remove single values and provide flags for each removed value?

### Conclusion

The main outcome is that PO-GRL algorithms (tested on COPOS and SAC) outperformed other algorithms (PPO, PPO-LSTM, TRPO, COPOS, SAC) in the Partially Observable Tasks
