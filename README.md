## Building NMT Feature function

```
cd ./moses/FF/NMT
make
cd -
```
This command builds plugin from amuNMT repository, which is a shared library. That means, you have to add an entry to LD_LIBRARY_PATH:
```
export LD_LIBRARY_PATH=$(PWD)/moses/FF/NMT/amunmt/build/src:$LD_LIBRARY_PATH
```

Adding NMT-FF to moses.ini
```
NeuralScoreFeature name=NMT0 mode=rescore config-path=./baseline/model/model.npz.amun.yml state-length=1
```

 * Only ``rescore`` mode is handled now.
 * **config-path** is the path to amunmt config file.
 * **state-length** must be 1
