# Document of MP_TRAINER(model-parallel trainer)

## Introduction

Speaker Verification is a large scale classification problem. Typically the number of speakers can reach 10000+ in an industrial level speaker verification system. A normal data parallel trainer cannot fit the need to train a large scale speaker verifcaiton model like 1000000+ identities. Model Parallel + Data Parallel is usually a good way to solve this problem. [Partial FC](https://arxiv.org/abs/2010.05222) is quite a good idea, however, the code implemented based on PyTorch needs to be carefully designed. To get rid of these complex codes, this repo adopted [ColossalAI](https://github.com/hpcaitech/ColossalAI) as model prallel trainer. Remember this project is still in experiments which means its instability.