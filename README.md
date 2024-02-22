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
![original](https://github.com/giorgioskij/tripod/assets/46194839/2db4c026-4528-4349-9eef-1f64ab0c54e2)



