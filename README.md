# jellyfish

海月 meets DQN!!

for relaxation

## Requirements
* tensorflow
- keras
- pygame
- numpy
- matplotlib

## Overview
ぶつかったりふえたりするげーむ

白いのがGAで進化したり  
赤いのが狩りを覚えるのをみてなごもう
![alt text](https://github.com/kitigai/jellyfish/blob/master/jellyfish_samp.png "Logo Title Text 1")


## Files
* RLball.py
- RLball2.py  
    * Policy Networkで学習
    * 軽い
    * おバカ

* dqn_ball.py
    * DQNで学習
    * 動かない。。。

* ballenv.py
    * dqn_ball用にenv化したクラゲゲーム
    * pygameのせいか描写がslow
