# ⚙️ Install PointNet


📣 **Make sure to use the same Python and CUDA versions as your application.**  In case you can't install or load CUDA to your environment, we provide a template DockerFile of a Ubuntu image with CUDA toolkit. We tested REGRACE on Python 3.11 and CUDA 11.7.

### [OPTIONAL] Using Docker

Build and run the docker container running:

```
docker run --runtime=nvidia --gpus all --mount type=bind,src="$(pwd)",target=/workspace -w /workspace -it $(docker build -q -f packages/pointnet2-wheel/Dockerfile .)
```


### Building the wheel

> This report requires the package `setuptools` in the version `69.*` or higher, but not `70.*`. You need CUDA installed in your machine to compile locally.

To compile the `pointnet2` wheel, run:
```bash
cd packages/pointnet2-wheel
python3.11 setup.py bdist_wheel
```

**Activiate your .venv** and  install the wheel:
```bash
pip install dist/pointnet2-0.0.0-cp<XXX>-cp<XXX>-<PLATFORM>.whl
```
for your Python version and platform.