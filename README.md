# CG & 3D Neural Renderer

This repo uses the PyTorch implementation of the paper [Neural 3D Mesh Renderer](http://hiroharu-kato.com/projects_en/neural_renderer.html) by Hiroharu Kato, Yoshitaka Ushiku, and Tatsuya Harada.
It uses a porting of the [original Chainer implementation](https://github.com/hiroharu-kato/neural_renderer) released by the authors.
The Python porting is realized by [daniilidis-group](https://github.com/daniilidis-group/neural_renderer).

The goal of this project is to combine the provided examples by daniilidis to have a generalized render pipile for 3D model based on [CelebMask-HQ dataset](https://github.com/switchablenorms/CelebAMask-HQ/blob/master/README.md).

There is a presentation of our work in this repository: [Presentation.pdf](https://github.com/iacopoerpichini/cg3d-neural-renderer/blob/master/Presentation.pdf).

The presentation contains some gifs and is possible to see it with pdf viewer like Adobe Acrobat Reader.
## Requirements
| Software                                                 | Version         |
| ---------------------------------------------------------|-----------------|
| **Python**                                               |     tested on v3.6    | 
| **pytorch** | tested on v1.2.0 |
| **yacs** | tested on v0.1.8 |
| **neural-renderer-pytorch** | tested on v1.1.3  |
| **scikit-image** | tested on v0.17.2  |

Python 3.6 and PyTorch 1.2.0.

**Note from daniilidis-group**: In some newer PyTorch versions you might see some compilation errors involving AT_ASSERT. In these cases you can use the version of the code that is in the branch *at_assert_fix*. These changes will be merged into master in the near future.

## Proposed Render Pipeline

This pipeline is our idea to render a 2D image on a 3D model (bfm_2009 [download](https://faces.dmi.unibas.ch/bfm/index.php?nav=1-2&id=downloads)).

<img src="https://github.com/iacopoerpichini/cg3d-neural-renderer/blob/master/data/pipeline.png" height="360" width="1200">

First we want to optimize the camara parameters respect to the nose, mouth and skin about a face, so we morph the obtained object a little for have a better countour and finally we apply the texture on the model.

### Directories Layout

This is the structure of our project
```bash
├── data                      # Data input output folder
│   ├── bfm-2009              # Put here 01_MorphableModel.mat and face05_4seg.mat
│   ├── bfm-2017              # Put here model2017-1_bfm_nomouth.h5
│   ├── ex-1                  # Example folder 
│   ├── ex-2
│   ├── ex-3
│   ├── ex-4
│   ├── tmp                   # Image for result gif
│   ├── out                   # Result of rendering
├── mesh                      
│   ├── read_bfm.py           # Read the bfm model of 2017
│   ├── read_bfm_2009.py      # Read the bfm 2009 model that includes segmentation point 
├── models                    # Contains all the three model used in the pipeline
│   ├── model_camera.py
│   ├── model_morphing.py
│   ├── model_textures.py
├── main.py                   # Core of experiments
├── config.py                 # Set experiment parameters
├── ...                       
```

In bfm-2009 and in bfm-2017 put the file downloaded by [BFM](https://faces.dmi.unibas.ch/bfm/index.php?nav=1-2&id=downloads)

## Run experiments
To run experiments open a terminal and run: ```phython main.py --example_input ex-1```

It is possible to change the experiment by command line and it's also possible to modify all the experimental parameters in the file config.py

## Results

Output produced:

![Alt Text](https://github.com/iacopoerpichini/cg3d-neural-renderer/blob/master/data/out/camera.gif)
![Alt Text](https://github.com/iacopoerpichini/cg3d-neural-renderer/blob/master/data/out/morphing.gif)
![Alt Text](https://github.com/iacopoerpichini/cg3d-neural-renderer/blob/master/data/out/rendered.gif)

First we can see the camera optimization referred to silhouettes of skin, nose and mouth in the ex-* folder, so we have the frontal view of the countour morphing and finally the render with the textures.
