
# char-rnn-korean
이 프로젝트는 문자 단위의 언어 모델을 훈련/샘플링하기 위해 **멀티 레이어 RNN(Recurrent Neural Network)**를 구현한 [char-rnn](https://github.com/karpathy/char-rnn)을 한글어를 지원하도록 확장한 코드다.

## 한글 지원

Lua에서 문자는 1바이트로 처리되며, 유니코드를 공식적으로 지원하지 않는다. 한글 언어는 유니코드를 사용하여,한 문자당 3바이트를 사용하므로 Lua의 문자열 처리함수를 사용할 수 없다. 이 프로젝트에서는 utf-8 지원 모듈인 [luautf8](https://github.com/starwing/luautf8)을 이용하여 [char-rnn](https://github.com/karpathy/char-rnn)이 한글 및 기타 유니코드 기반 언어에서 동작하도록 확장했다. 아래는 원본을 번역한 내용과 몇가지 주석을 덧붙였다.

원본 Torch 코드는 그대로 두었으며, 한글어를 지원하도록 변경한 코드에는 파일명에 `kor`를 추가했다. 아래는 변경한 파일 목록이다.
* util/CharKorSplitLMMinibatchLoader.lua
** util/CharSplitLMMinibatchLoader.lua 코드에서 데이터셋을 처리하는 부분에 utf-8을 지원하도록 변경
* train_kor.lua
** train.lua 훈련 코드에서 CharSplitLMMinibatchLoader 대신 CharKorSplitLMMinibatchLoader을 사용하도록 변경
* sample_kor.lua
** sample.lua 샘플링 코드에서 primetext(생성할 텍스트의 앞부분)을 처리하는 코드에서 utf-8을 지원하도록 변경 

## 개요

이 코드는 문자 단위의 언어 모델을 훈련/샘플링하기 위해 **멀티 레이어 Recurrent Neural Network** (RNN, LSTM, GRU) 구현한다. 다시 말해 하나의 텍스트 파일을 입력으로 받아 시퀀스의 다음 문자를 예측하도록 RNN을 훈련시킨다. 따라서 학습된 RNN 모델을 사용하면 문자 단위로 원본 훈련 데이터와 유사한 텍스트를 생성할 수 있다. 왜 이 코드를 만들었는지에 대한 내용은 내 블로그의 [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)에서 확인할 수 있다.

Torch/Lua/신경망을 처음 접한다면, 파이썬/Numpy로 작성했던 [100 라인 gist](https://gist.github.com/karpathy/d4dee566867f8291f086) 코드를 좀 더 멋지게 다듬은 버전이므로 해당 코드를 보면 도움이 될 것이다. 여기 코드에서는 아래와 같은 기능을 더 추가했다.
* 멀티 레이어 가능
* 기본 RNN 뿐만 아니라 LSTM을 사용 가능
* 모델 체크포인트를 위한 추가적인 지원 기능
* 미니 배치와 GPU를 사용할 수 있으므로 더 효율적

## 변경사항: torch-rnn

최근 [Justin Johnson](http://cs.stanford.edu/people/jcjohns/) (@jcjohnson)는 더 멋진/적은/깔끔한/빠른 Torch 코드를 기반으로 char-rnn을 처음부터 다시 구현했다. 이 코드는 [torch-rnn](https://github.com/jcjohnson/torch-rnn)에서 확인할 수 있다. 최적화에는 Adam을 사용하며, RNN/LSTM의 포워드/백워드 과정을 직접 구현하여 메모리/시간 효율성을 높였다. 또한 여기 코드에서 모델을 복사(clone)하는 복잡한 부분을 피할 수 있다. 따라서 char-rnn 구현체가 필요하다면 여기에 구현된 코드보다는 앞으로는 torch-rnn을 기본으로 사용해야 한다.
> 주) 최근 1년간 이 레파지토리에 변경사항은 없는 상태다. 다음에는 torch-rnn이 한글을 지원하도록 확장하겠다. 코드는 오래되었더라도, 저자의 블로그 글인 [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)과 함께 이 레파지토리의 코드를 따라해보면 나와 같이 RNN을 처음 접하는 사용자들에게는 꽤 도움이 될 것이다.

## 요구사항

이 코드는 Lua를 이용하여 작성했으며 [Torch](http://torch.ch/)를 필요로 한다. 우분투를 사용한다면, 아래와 같이 Torch를 홈 디렉토리에 설치할 수 있다:

```bash
$ curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
$ git clone https://github.com/torch/distro.git ~/torch --recursive
$ cd ~/torch; 
$ ./install.sh      # and enter "yes" at the end to modify your bashrc
$ source ~/.bashrc
```

더 자세한 내용은 Torch 설치 문서를 참고한다. Torch를 설치한 후에는 [LuaRocks](https://luarocks.org/)를 이용하여 몇가지 패키지를 추가로 설치한다(LuaRocks는 앞의 방법으로 Torch를 설치하면 기본으로 설치된다). 아래는 필수적이다.

```bash
$ luarocks install nngraph 
$ luarocks install optim
$ luarocks install nn
```

NVIDIA GPU 기반으로 CUDA를 사용하여 학습하려면(GPU보다 15배 빠르다), 다연히 GPU가 있어야 하며, [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)도 설치해야 한다. 그리고 `cutorch`와 `cunn` 패키지도 설치한다.

```bash
$ luarocks install cutorch
$ luarocks install cunn
```

또는 OpenCL GPU(ATI 카드 등)를 사용하려면, 대신에 `cltorch`와 `clnn`를 설치해야 하며, 학습시킬 때 `-opencl 1` 옵션을 사용해야 한다 ([cltorch issues](https://github.com/hughperkins/cltorch/issues)).

```bash
$ luarocks install cltorch
$ luarocks install clnn
```

## 사용법

### 데이터

입력 데이터는 모두 `data/` 디렉토리에 저장해야 한다. 레파지토리에는 하나의 예제 데이터셋이 포함되어 있는데(`data/tinyshakespeare`에 위치), 세익스피어의 작품 중 일부를 포함하고 있다. 몇가지 추가적인 데이터셋은 [여기](http://cs.stanford.edu/people/karpathy/char-rnn/)에서 제공한다.

**자신만의 데이터**: 자신만의 데이터를 직접 사용하려면 `input.txt`라는 이름의 단일 파일을 만들어서 `data/` 디렉토리의 폴더 안에 위치시킨다. 예를 들면 `data/some_folder/input.txt`과 같다. 훈련 스크립트를 처음 실행시키는 경우에 입력 데이터에 대해 몇가지 전처리 작업을 거친 후 편의를 위해 2개의 캐시 파일을 `data/some_folder`에 생성한다.

**데이터셋 사이즈**: 데이터 사이즈가 너무 작다면(예를 들어 1MB는 너무 작다), RNN은 효과적으로 학습되지 않는다. RNN은 아무것도 없는 상태에서 모든 것을 학습한다는 사실을 염두에 두어야 한다. 반대로 데이터가 너무 크다면(예를 드러 2MB 이상), 고민하지 말고 `rnn_size`를 높여서 더 큰 모델을 훈련시킨다 (더 자세한 내용은 아래에서 설명한다). 모델은 *상당히 개선*될 것이다. 예를 들어 데이터가 6MB라면 `rnn_size`을 300 또는 그 이상으로 높인다. 나의 GPU에서 이 코드를 이용하여 학습했을 때 `num_layers`가 3인 경우(기본값은 2) 처리할 수 있었던 가장 큰 `rnn_size`는 700이었다.

### 훈련

모델을 훈련하려면 `train.lua`를 사용한다.

Start training the model using `train.lua`. As a sanity check, to run on the included example dataset simply try:

```
$ th train.lua -gpuid -1
```

Notice that here we are setting the flag `gpuid` to -1, which tells the code to train using CPU, otherwise it defaults to GPU 0.  There are many other flags for various options. Consult `$ th train.lua -help` for comprehensive settings. Here's another example that trains a bigger network and also shows how you can run on your own custom dataset (this already assumes that `data/some_folder/input.txt` exists):

```
$ th train.lua -data_dir data/some_folder -rnn_size 512 -num_layers 2 -dropout 0.5
```

**Checkpoints.** While the model is training it will periodically write checkpoint files to the `cv` folder. The frequency with which these checkpoints are written is controlled with number of iterations, as specified with the `eval_val_every` option (e.g. if this is 1 then a checkpoint is written every iteration). The filename of these checkpoints contains a very important number: the **loss**. For example, a checkpoint with filename `lm_lstm_epoch0.95_2.0681.t7` indicates that at this point the model was on epoch 0.95 (i.e. it has almost done one full pass over the training data), and the loss on validation data was 2.0681. This number is very important because the lower it is, the better the checkpoint works. Once you start to generate data (discussed below), you will want to use the model checkpoint that reports the lowest validation loss. Notice that this might not necessarily be the last checkpoint at the end of training (due to possible overfitting).

Another important quantities to be aware of are `batch_size` (call it B), `seq_length` (call it S), and the `train_frac` and `val_frac` settings. The batch size specifies how many streams of data are processed in parallel at one time. The sequence length specifies the length of each stream, which is also the limit at which the gradients can propagate backwards in time. For example, if `seq_length` is 20, then the gradient signal will never backpropagate more than 20 time steps, and the model might not *find* dependencies longer than this length in number of characters. Thus, if you have a very difficult dataset where there are a lot of long-term dependencies you will want to increase this setting. Now, if at runtime your input text file has N characters, these first all get split into chunks of size `BxS`. These chunks then get allocated across three splits: train/val/test according to the `frac` settings. By default `train_frac` is 0.95 and `val_frac` is 0.05, which means that 95% of our data chunks will be trained on and 5% of the chunks will be used to estimate the validation loss (and hence the generalization). If your data is small, it's possible that with the default settings you'll only have very few chunks in total (for example 100). This is bad: In these cases you may want to decrease batch size or sequence length.

Note that you can also initialize parameters from a previously saved checkpoint using `init_from`.

### Sampling

Given a checkpoint file (such as those written to `cv`) we can generate new text. For example:

```
$ th sample.lua cv/some_checkpoint.t7 -gpuid -1
```

Make sure that if your checkpoint was trained with GPU it is also sampled from with GPU, or vice versa. Otherwise the code will (currently) complain. As with the train script, see `$ th sample.lua -help` for full options. One important one is (for example) `-length 10000` which would generate 10,000 characters (default = 2000).

**Temperature**. An important parameter you may want to play with is `-temperature`, which takes a number in range \(0, 1\] (0 not included), default = 1. The temperature is dividing the predicted log probabilities before the Softmax, so lower temperature will cause the model to make more likely, but also more boring and conservative predictions. Higher temperatures cause the model to take more chances and increase diversity of results, but at a cost of more mistakes.

**Priming**. It's also possible to prime the model with some starting text using `-primetext`. This starts out the RNN with some hardcoded characters to *warm* it up with some context before it starts generating text. E.g. a fun primetext might be `-primetext "the meaning of life is "`. 

**Training with GPU but sampling on CPU**. Right now the solution is to use the `convert_gpu_cpu_checkpoint.lua` script to convert your GPU checkpoint to a CPU checkpoint. In near future you will not have to do this explicitly. E.g.:

```
$ th convert_gpu_cpu_checkpoint.lua cv/lm_lstm_epoch30.00_1.3950.t7
```

will create a new file `cv/lm_lstm_epoch30.00_1.3950.t7_cpu.t7` that you can use with the sample script and with `-gpuid -1` for CPU mode.

Happy sampling!

## Tips and Tricks

### Monitoring Validation Loss vs. Training Loss
If you're somewhat new to Machine Learning or Neural Networks it can take a bit of expertise to get good models. The most important quantity to keep track of is the difference between your training loss (printed during training) and the validation loss (printed once in a while when the RNN is run on the validation data (by default every 1000 iterations)). In particular:

- If your training loss is much lower than validation loss then this means the network might be **overfitting**. Solutions to this are to decrease your network size, or to increase dropout. For example you could try dropout of 0.5 and so on.
- If your training/validation loss are about equal then your model is **underfitting**. Increase the size of your model (either number of layers or the raw number of neurons per layer)

### Approximate number of parameters

The two most important parameters that control the model are `rnn_size` and `num_layers`. I would advise that you always use `num_layers` of either 2/3. The `rnn_size` can be adjusted based on how much data you have. The two important quantities to keep track of here are:

- The number of parameters in your model. This is printed when you start training.
- The size of your dataset. 1MB file is approximately 1 million characters.

These two should be about the same order of magnitude. It's a little tricky to tell. Here are some examples:

- I have a 100MB dataset and I'm using the default parameter settings (which currently print 150K parameters). My data size is significantly larger (100 mil >> 0.15 mil), so I expect to heavily underfit. I am thinking I can comfortably afford to make `rnn_size` larger.
- I have a 10MB dataset and running a 10 million parameter model. I'm slightly nervous and I'm carefully monitoring my validation loss. If it's larger than my training loss then I may want to try to increase dropout a bit and see if that heps the validation loss.

### Best models strategy

The winning strategy to obtaining very good models (if you have the compute time) is to always err on making the network larger (as large as you're willing to wait for it to compute) and then try different dropout values (between 0,1). Whatever model has the best validation performance (the loss, written in the checkpoint filename, low is good) is the one you should use in the end.

It is very common in deep learning to run many different models with many different hyperparameter settings, and in the end take whatever checkpoint gave the best validation performance.

By the way, the size of your training and validation splits are also parameters. Make sure you have a decent amount of data in your validation set or otherwise the validation performance will be noisy and not very informative.

## Additional Pointers and Acknowledgements

This code was originally based on Oxford University Machine Learning class [practical 6](https://github.com/oxford-cs-ml-2015/practical6), which is in turn based on [learning to execute](https://github.com/wojciechz/learning_to_execute) code from Wojciech Zaremba. Chunks of it were also developed in collaboration with my labmate [Justin Johnson](http://cs.stanford.edu/people/jcjohns/).

To learn more about RNN language models I recommend looking at:

- [My recent talk](https://skillsmatter.com/skillscasts/6611-visualizing-and-understanding-recurrent-networks) on char-rnn
- [Generating Sequences With Recurrent Neural Networks](http://arxiv.org/abs/1308.0850) by Alex Graves
- [Generating Text with Recurrent Neural Networks](http://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf) by Ilya Sutskever
- [Tomas Mikolov's Thesis](http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf)

## License

MIT
