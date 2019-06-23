#### ARA-CNN

This repository contains an implementation of ARA-CNN - a Bayesian deep learning model intended for histopathological image classification.

ARA stands for Accurate, Reliable and Active. Any method that works on patient data needs to be accurate and reliable, meaning that in addition to very high classification accuracy it should also provide a measure of uncertainty for each prediction. ARA-CNN adheres to these requirements. Moreover, the uncertainty measurement can be used as an acquisition function in active learning, which significantly speeds up the learning process on new histopathological datasets.

##### Installation

Required dependencies: Python 3.6+

Installation process:

```
git clone https://github.com/animgoeth/pathology-bayesian-cnn.git
cd pathology-bayesian-cnn
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

##### Dataset

Your dataset needs to have the following directory structure:

```
-+ dataset
--+ train
---+ class_0
---+ class_1
⋮
---+ class_N
--+ test
---+ class_0
---+ class_1
⋮
---+ class_N
```

Each `class_i` directory should contain image patches.

Modify `CLASS_DICT` in `src/config.py` to match your dataset's classes.

##### Training

Execute the following command:

`python src/simple_cnn.py --output-path /your/output/path --dataset-path /your/dataset/path`

This command starts the training process on data in `/your/dataset/path/train`. After training, it performs inference on images in `/your/dataset/path/test` and writes all results to `/your/output/path/ara_cnn.txt` and the trained model to `/your/output/path/ara_cnn.h5`.

`ara_cnn.txt` has the following format:

```
epoch_nr, loss, main_output_loss, aux_output_loss, main_output_acc, aux_output_acc, val_loss, val_main_output_loss, val_aux_output_loss, val_main_output_acc, val_aux_output_acc, eval_main_acc, eval_aux_acc
1, 2.064821, 1.992539, 1.831265, 0.398278, 0.324298, 4.835865, 4.972194, 2.721522, 0.196006, 0.137574, 0.906250, 0.893229
⋮
100, 0.481360, 0.392671, 0.464361, 0.867985, 0.844388, 0.359386, 0.278197, 0.274887, 0.909024, 0.897929, 0.906250, 0.893229
```

##### Testing

Execute the following command:

`python src/test_model.py --input-images /your/input/images/path --model-path /your/model/path/ara_cnn.h5 --output-path /your/output/path --measure MEASURE`

All images in `/your/input/images/path` are tested and after that both the classification and uncertainty results are saved to `/your/output/path/results.csv`. The uncertainty is measured with the provided `MEASURE` - either `Entropy` or `BALD`.

`results.csv` has the following format:
```
image_name, uncertainty, class_0_probablity, ..., class_N_probability
img1.jpg, 0.1, 0.9, ..., 0.01
⋮
imgX.jpg, 0.9, 0.3, ..., 0.2
```

Note: In order to make the testing process work, you need to modify one of the Keras source files. In `venv/lib/python3.x/site-packages/keras/layers/advanced_activations.py` replace line 38 `self.alpha = K.cast_to_floatx(alpha)` with the following code:

```
try:
    self.alpha = K.cast_to_floatx(alpha)
except TypeError:
    self.alpha = K.cast_to_floatx(alpha['value'])
```

##### Paper

The ARA framework and all accompanying results are described in this pre-print paper:

_**ARA: accurate, reliable and active histopathological image classification framework with Bayesian deep learning**_
<br />
Łukasz Rączkowski, Marcin Możejko, Joanna Zambonelli, Ewa Szczurek
<br />
bioRxiv 658138; doi: https://doi.org/10.1101/658138 

##### Contact

For more information, contact one of the following:
- l.raczkowski@mimuw.edu.pl
- mmozejko1988@gmail.com
- eszczurek@mimuw.edu.pl
