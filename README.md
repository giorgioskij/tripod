# Tripod

![logo](https://github.com/giorgioskij/tripod/assets/46194839/d68dd387-bd69-442f-a948-98351d3d0ed2)

This is Tripod, an AI Image Sharpener. It tries its best to remove or reduce **motion blur** from pictures. 

Feel free to test it, use it, improve it.

## Status
Right now, exetremely minimal usecases are covered:
- Tripod is only meant to receive as input square images with sizes that are powers of 2 (so 256x256, 512x512, 1024x1024). Crop your images accordingly before using it, or be prepared to read error logs.
- Tripod was trained on images that are artificially blurred. The guy is trying his best, don't expect him to perfectly sharpen a real-world blurry image.

## Install

### Clone the repo
```
git clone https://github.com/giorgioskij/tripod
cd tripod
```
### Install the dependencies
```
conda env create -f environment.yml
conda activate tripod
```

## Use 
The best way to use Tripod is to interact with the code directly. However, if you just want to test the model, you can run
```
python scripts/demo.py <path to your image>
```
The sharpened image should be saved in the same directory as the input one.

## Examples
![35output](https://github.com/giorgioskij/tripod/assets/46194839/96c1b0a1-28e6-42fb-b742-5060eb6d16e1)
![35input](https://github.com/giorgioskij/tripod/assets/46194839/98d83dd9-0230-413f-aab3-56261919d231)


![91output](https://github.com/giorgioskij/tripod/assets/46194839/e14429df-421f-4d1a-a448-5536bf8750cf)
![91input](https://github.com/giorgioskij/tripod/assets/46194839/8be7be47-00cb-40ce-a7bd-0095f236ce7a)


