## Recent Improvements

### Enhanced Inference and Model Comparison

The repository now includes improved inference capabilities and model comparison tools:

#### Inference Improvements (`inference.py`)
- Added support for multiple model architectures:
  - RCAN (Residual Channel Attention Network)
  - RealPLKSR (Real-world Partial Large Kernel SR)
  - ArtCNN (with configurations R8F48, R8F64, R16F96)
  - Compact (VGG-style network with efficient design)
- Optimized memory handling with tiling for large images
- Automatic device selection (CUDA, MPS, CPU) with fallback options
- Smooth tile blending with configurable overlap
- Support for both safetensors and PyTorch model formats

Usage:
```bash
# Basic usage
python inference.py --model pretrained_model/model_name.safetensors --input input.png --output output.png

# Advanced usage with tiling options
python inference.py --model pretrained_model/model_name.safetensors --input input.png --output output.png --tile_size 512 --overlap 32 --device mps
```

#### Model Comparison Tool (`compare_models.py`)
- Comprehensive model comparison with multiple metrics:
  - PSNR (Peak Signal-to-Noise Ratio)
  - SSIM (Structural Similarity Index)
  - LPIPS (Learned Perceptual Image Patch Similarity)
  - Sharpness ratio
- Automatic processing with all available models
- Detailed JSON reports and human-readable summaries
- Visual comparison of results
- Model performance ranking

Usage:
```bash
# Compare all models in the pretrained_model directory
python compare_models.py --input image.png --models_dir ./pretrained_model --output_dir ./results

# Specify device explicitly
python compare_models.py --input image.png --models_dir ./pretrained_model --output_dir ./results --device mps
```

The comparison tool generates:
- Upscaled images from each model
- Detailed metrics in JSON format
- Human-readable summary with model rankings
- Top 3 models comparison with scores

Results are saved in a timestamped directory under the specified output directory, containing:
- Upscaled images from each model
- `comparison_report.json` with detailed metrics
- `summary.txt` with human-readable results

### Hardware Support
- NVIDIA GPUs (CUDA)
- Apple Silicon (MPS)
- AMD GPUs (ROCm on Linux)
- CPU fallback for all platforms

These improvements make it easier to:
1. Process images with multiple state-of-the-art models
2. Compare model performance objectively
3. Choose the best model for specific use cases
4. Handle large images efficiently
5. Utilize available hardware acceleration

The result is: (on the image that previously been processed by hugging face space upscaler)

Top 3 Models:
1. 2x_DF2K_ArtCNN_R8F48_450k
   Score: 0.9493
   PSNR: 42.26
   SSIM: 0.9948
   LPIPS: 0.0135
2. 2x_DF2K_ArtCNN_R8F64_450k
   Score: 0.9493
   PSNR: 42.26
   SSIM: 0.9948
   LPIPS: 0.0135
3. 2x_DF2K_Redux_Compact_450k
   Score: 0.9490
   PSNR: 42.23
   SSIM: 0.9947
   LPIPS: 0.0137


# traiNNer-redux
![redux3](https://github.com/user-attachments/assets/d107b2fc-6b68-4d3e-b08d-82c8231796cb)

## Overview
[traiNNer-redux](https://trainner-redux.readthedocs.io/en/latest/index.html) is a deep learning training framework for image super resolution and restoration which allows you to train PyTorch models for upscaling and restoring images and videos. NVIDIA graphics card is recommended, but AMD works on Linux machines with ROCm.

## Usage Instructions
Please see the [getting started](https://trainner-redux.readthedocs.io/en/latest/getting_started.html) page for instructions on how to use traiNNer-redux.

## Contributing
Please see the [contributing](https://trainner-redux.readthedocs.io/en/latest/contributing.html) page for more info on how to contribute.

## Resources
- [OpenModelDB](https://openmodeldb.info/): Repository of AI upscaling models, which can be used as pretrain models to train new models. Models trained with this repo can be submitted to OMDB.
- [chaiNNer](https://github.com/chaiNNer-org/chaiNNer): General purpose tool for AI upscaling and image processing, models trained with this repo can be run on chaiNNer. chaiNNer can also assist with dataset preparation.
- [WTP Dataset Destroyer](https://github.com/umzi2/wtp_dataset_destroyer): Tool to degrade high quality images, which can be used to prepare the low quality images for the training dataset.
- [helpful-scripts](https://github.com/Kim2091/helpful-scripts): Collection of scripts written to improve experience training AI models.
- [Enhance Everything! Discord Server](https://discord.gg/cpAUpDK): Get help training a model, share upscaling results, submit your trained models, and more.
- [vs_align](https://github.com/pifroggi/vs_align): Video Alignment and Synchonization for Vapoursynth, tool to align LR and HR datasets.
- [ImgAlign](https://github.com/sonic41592/ImgAlign): Tool for auto aligning, cropping, and scaling HR and LR images for training image based neural networks.

## License and Acknowledgement

traiNNer-redux is released under the [Apache License 2.0](LICENSE.txt). See [LICENSE](LICENSE/README.md) for individual licenses and acknowledgements.

- This repository is a fork of [joeyballentine/traiNNer-redux](https://github.com/joeyballentine/traiNNer-redux) which itself is a fork of [BasicSR](https://github.com/XPixelGroup/BasicSR).
- Network architectures are imported from [Spandrel](https://github.com/chaiNNer-org/spandrel).
- Several architectures are developed by [umzi2](https://github.com/umzi2): [ArtCNN-PyTorch](https://github.com/umzi2/ArtCNN-PyTorch), [DUnet](https://github.com/umzi2/DUnet), [FlexNet](https://github.com/umzi2/FlexNet), [MetaGan](https://github.com/umzi2/MetaGan), [MoESR](https://github.com/umzi2/MoESR), [MoSR](https://github.com/umzi2/MoSR), [RTMoSR](https://github.com/rewaifu/RTMoSR), [SPANPlus](https://github.com/umzi2/SPANPlus)
- The [ArtCNN](https://github.com/Artoriuz/ArtCNN) architecture is originally developed by [Artoriuz](https://github.com/Artoriuz).
- The TSCUNet architecture is from [aaf6aa/SCUNet](https://github.com/aaf6aa/SCUNet) which is a modification of [SCUNet](https://github.com/cszn/SCUNet), and parts of the training code for TSCUNet are adapted from [TSCUNet_Trainer](https://github.com/Demetter/TSCUNet_Trainer).
- Several enhancements reference implementations from [Corpsecreate/neosr](https://github.com/Corpsecreate/neosr) and its original repo [neosr](https://github.com/muslll/neosr).
- Members of the Enhance Everything Discord server: [Corpsecreate](https://github.com/Corpsecreate), [joeyballentine](https://github.com/joeyballentine), [Kim2091](https://github.com/Kim2091).
