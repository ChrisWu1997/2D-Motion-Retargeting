Here we provide some scripts and instructions to reproduce important figures in the paper.

Besides some shell scripts listed in this folder, many of other figures are generated from videos, which are shared at [here](https://drive.google.com/drive/folders/1ioxFNknkOz6pVdxpp1CaUvKtlJvfe1SU?usp=sharing).
The following figs use the results that come from two source videos:
>Fig 1, fig 2 and fig 5: ballet_woman.mp4 + breakdance_man.mp4  
>Fig 13(b): shortman.mp4 + penn0067.mp4   
>Fig 16(a): dancer-forward.mp4 + model-1-2.mp4  
>Fig 16(b): workout-jumpingjack.mp4 + ballet_women_roll.mp4

To reproduce the above figures from source videos, you need to first use [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to extract the 2D joint positions and then run our model as suggested in the [README](https://github.com/ChrisWu1997/2D-Motion-Retargeting#run-demo-examples).
