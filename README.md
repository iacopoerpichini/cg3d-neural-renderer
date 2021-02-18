# CG & 3D Neural Renderer
## Application of human face images on 3D head models using machine learning techniques

The aim of the project is to apply images of human faces to a 3D head model. Machine learning techinques are used to adapt the model to the face image and create a convincing result.

The project uses the Neural Renderer framework first proposed in the paper [Neural 3D Mesh Renderer](http://hiroharu-kato.com/projects_en/neural_renderer.html) by Hiroharu Kato, Yoshitaka Ushiku, and Tatsuya Harada. The Neural Renderer implements a rendering pipeline that can be embedded into a trainable deep learning model.
More specifically, its PyTorch implementation realized by [daniilidis-group](https://github.com/daniilidis-group/neural_renderer) is used.

A presentation that focuses on the details of this work can be found here: [Presentation.pdf](https://github.com/iacopoerpichini/cg3d-neural-renderer/blob/master/Presentation.pdf).
To correctly view the contained gifs it is recommended to open it with Adobe Acrobat Reader.

## Requirements
| Software                                                 | Version         |
| ---------------------------------------------------------|-----------------|
| **Python**                                               |     tested on v3.6    | 
| **pytorch** | tested on v1.2.0 |
| **yacs** | tested on v0.1.8 |
| **neural-renderer-pytorch** | tested on v1.1.3  |
| **scikit-image** | tested on v0.17.2  |

**Note from daniilidis-group**: In some newer PyTorch versions you might see some compilation errors involving AT_ASSERT. In these cases you can use the version of the code that is in the branch *at_assert_fix*. These changes will be merged into master in the near future.

## Data

- 3D model: BFM 2009 ([download](https://faces.dmi.unibas.ch/bfm/index.php?nav=1-2&id=downloads))
- Face images: CelebAMask-HQ ([official repository](https://github.com/switchablenorms/CelebAMask-HQ))

## Proposed Render Pipeline

The project is based on the following 5-step pipeline:

<img src="https://github.com/iacopoerpichini/cg3d-neural-renderer/blob/master/data/pipeline.png" height="300" width="1200">

After some preprocessing operations, the application optimizes the camera position from which the model will be rendered and the 3D mesh vertices to fit the target face silhouette. Then, the model textures are optimized to match the face image. Finally, a rendering of the calculated 3D model is produced from multiple points of view.

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
