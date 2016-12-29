
# char-rnn-korean
이 프로젝트는 문자 단위의 언어 모델을 훈련/샘플링하기 위해 **멀티 레이어 RNN(Recurrent Neural Network)**를 구현한 [char-rnn](https://github.com/karpathy/char-rnn)을 한글어를 지원하도록 확장한 코드다.

## 한글 지원

Lua에서 문자는 1바이트로 처리되며, 유니코드를 공식적으로 지원하지 않는다. 한글 언어는 유니코드를 사용하여,한 문자당 3바이트를 사용하므로 Lua의 문자열 처리함수를 사용할 수 없다. 이 프로젝트에서는 utf-8 지원 모듈인 [luautf8](https://github.com/starwing/luautf8)을 이용하여 [char-rnn](https://github.com/karpathy/char-rnn)이 한글 및 기타 유니코드 기반 언어에서 동작하도록 확장했다. 아래는 원본을 번역한 내용과 몇가지 주석을 덧붙였다.

원본 Torch 코드는 그대로 두었으며, 한글어를 지원하도록 변경한 코드에는 파일명에 `kor`를 추가했다. 아래는 변경한 파일 목록이다.
* util/CharKorSplitLMMinibatchLoader.lua
  * util/CharSplitLMMinibatchLoader.lua 코드에서 데이터셋을 처리하는 부분에 utf-8을 지원하도록 변경
  * util/CharSplitLMMinibatchLoader.lua 코드에서 입력데이터셋을 ByteTensor에 저장하던 부분을 ShortTensor를 사용하도록 변경
* train_kor.lua
  * train.lua 훈련 코드에서 CharSplitLMMinibatchLoader 대신 CharKorSplitLMMinibatchLoader을 사용하도록 변경
* sample_kor.lua
  * sample.lua 샘플링 코드에서 primetext(생성할 텍스트의 앞부분)을 처리하는 코드에서 utf-8을 지원하도록 변경

또한 한글 데이터를 학습할 수 있도록 발라드 노래 가사 데이터셋을 **data/lyrics_ballad/input.txt`**에 포함했다.

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

### 한글 지원

`한글(utf-8)`을 지원하기 위해 [luautf8](https://github.com/starwing/luautf8) 패키지를 추가로 설치한다.

```bash
$ luarocks install luautf8
```

## 사용법

### 데이터

입력 데이터는 모두 `data/` 디렉토리에 저장해야 한다. 레파지토리에는 하나의 예제 데이터셋이 포함되어 있는데(`data/tinyshakespeare`에 위치), 세익스피어의 작품 중 일부를 포함하고 있다. 몇가지 추가적인 데이터셋은 [여기](http://cs.stanford.edu/people/karpathy/char-rnn/)에서 제공한다.

**자신만의 데이터**: 자신만의 데이터를 직접 사용하려면 `input.txt`라는 이름의 단일 파일을 만들어서 `data/` 디렉토리의 폴더 안에 위치시킨다. 예를 들면 `data/some_folder/input.txt`과 같다. 훈련 스크립트를 처음 실행시키는 경우에 입력 데이터에 대해 몇가지 전처리 작업을 거친 후 편의를 위해 2개의 캐시 파일을 `data/some_folder`에 생성한다.

**데이터셋 사이즈**: 데이터 사이즈가 너무 작다면(예를 들어 1MB는 너무 작다), RNN은 효과적으로 학습되지 않는다. RNN은 아무것도 없는 상태에서 모든 것을 학습한다는 사실을 염두에 두어야 한다. 반대로 데이터가 너무 크다면(예를 드러 2MB 이상), 고민하지 말고 `rnn_size`를 높여서 더 큰 모델을 훈련시킨다 (더 자세한 내용은 아래에서 설명한다). 모델은 *상당히 개선*될 것이다. 예를 들어 데이터가 6MB라면 `rnn_size`을 300 또는 그 이상으로 높인다. 나의 GPU에서 이 코드를 이용하여 학습했을 때 `num_layers`가 3인 경우(기본값은 2) 처리할 수 있었던 가장 큰 `rnn_size`는 700이었다.

### 훈련

모델을 훈련하려면 `train_kor.lua`를 사용한다. 제대로 설치되었는지 확인하기 위해, 예제 데이터셋을 이용하여 훈련하도록 아래를 실행해본다:

```
$ th train_kor.lua -gpuid -1
```

보다시피 여기에서는 `gpuid` 플래그를 -1로 설정했ㄴ느데, 이는 CPU를 사용하여 훈련하도록 만든다. 기본값인 GPU이며 0이다. 다양한 옵션을 지원하기 위해 많은 플래그가 있다. 이들 설정을 모두 확인하려면 `$ th train_kor.lua -help`을 실행한다. 예를 들어 아래의 경우에는 더 큰 네트워크를 학습하며, 자신만의 커스텀 데이터를 사용하는 방법을 설명한다(이 경우 데이터셋이 `data/some_folder/input.txt`에 위치해야 한다):

```
$ th train_kor.lua -data_dir data/some_folder -rnn_size 512 -num_layers 2 -dropout 0.5
```

**체크포인트.** 모델이 학습되는 도중에 주기적으로 체크포인트 파일을 `cv` 디렉토리에 생성한다. 이들 체크포인트가 생성되는 주기는 이터페이션 숫자에 따라 정해지며, `eval_val_every` 옵션으로 설정할 수 있다 (예를 들어 이 설정값이 1이라면, 체크포인트는 매 이터레이션마다 생성된다). 체크포인트 파일명에는 중요한 숫자값인 **loss**가 포함된다. 예를 들어 체크포인트 파일명이 `lm_lstm_epoch0.95_2.0681.t7`라면, 이 시점에서 모델은 0.95 에폭이었고(즉 훈련셋에 대해 첫 에폭이 거의 다 진행된 상태), 평가 데이터셋에 대한 손실은 2.0681이었다라는 점을 말해준다. 이 값은 매우 중요한데, 체크포인트 파일명의 손실값이 적을수록 더 제대로 동작하는 모델이기 때문이다. 텍스트를 생성해야 할 때(아래에서 설명한다), 평가 손실 값이 적은 체크포인트 모델을 사용하는 것이 좋다. 주의할 점은 훈련 단계의 마지막 체크포인트가 항상 가장 적은 손실값을 가지는 것은 아니라는 점이다(과적합이 발생하므로).

중요한 또 다른 설정은 `batch_size` (B라고 부르자), `seq_length` (S라고 부르자), `train_frac` and `val_frac`이다. `batch_size`는 병렬로 한꺼번에 처리할 데이터 스트립의 개수다. `seq_length`는 각 데이터 스트림의 길이로, 그래디언트가 시간축에 대해 back propagation할 수 있는 범위를 한정한다. 예를 들어 `seq_length`가 20이라면, 그래디언트 시그널은 시간축으로 20 단계를 초과하여 back propagation하지 않는다. 그리고 문자들의 길이가 이 크기를 벗어나면 모델은 문자간의 의존성을 **발견**하지 못한다. 따라서 데이터셋이 복잡하여 단어간의 의존성이 긴 길이이 걸쳐 있는 경우라면, `seq_length` 설정값을 높여야 한다. 이제 예를 들어 실행 단계에서 입력 텍스트 파일에 N개의 문자가 있다고 하면, 이들은 모두 `BxS` 크기의 청크들로 분할된다. 이어서 이들 청크는 `frac` 설정에 따라 3개의 split train/al/test에 각각 할당된다. 기본값은 `train_frac`의 경우 0.95, `val_frac`은 0.05이다. 즉 입력 데이터셋의 청크 중 96%는 훈련 과정에서, 5%는 평가 손실을 추정할 때 사용한다. 데이터가 작은 경우 이들 기본값을 그대로 사용하면 너무 적은 수의 청크가 생성된다(예를 들어 100개 정도). 이는 좋은 상태가 아니다. 데이터가 적다면, `batch_size`와 `seq_length`를 줄이는 편이 좋다.

`init_from` 설정을 통해, 이전에 저장한 체크포인트 파일을 이용하여 파라미터를 초기화할 수도 있다.

### 샘플링

체크포인트 파일을 이용하면(`cv` 디렉토리에 생성된), 새로운 텍스트를 생성할 수 있다. 예를 들어:

```
$ th sample_kor.lua cv/some_checkpoint.t7 -gpuid -1
```

체크포인트가 GPU를 이용하여 학습되었다면, 반드시 GPU를 이용하여 샘플링해야 하며, CPU도 마찬가지다. 그렇지 않은 경우 코드는 (현재) 경고를 출력한다. 훈련 스크립트와 마찬가지로, `$ th sample.lua -help`을 입력하면 전체 옵션을 확인할 수 있다. 이 중 중요한 옵션은(예를 들어) `-length 10000`인데, 이 경우 10,000문자 길이의 텍스트를 생성한다(기본값은 2000이다).

**Temperature**. 가지고 놀만한 중요한 파라미터는 `-temperature`로, \(0, 1\](0 제외) 사이의 숫자값이며, 기본값은 1이다. Softmax 전에 예측된 로그 확률값을 temperature로 나눈다. 따라서 temperature가 작을수록 모델은 그럴법한 텍스트를 생성하지만, 예측된 결과는 다소 지루하고 보수적이다. temperature가 높다면 더 많은 우연을 포함하며 결과가 더 다양해지지만, 오류가 더 많이 포함된다.

**Priming**. 모델에 사전 지식을 주는 것도 가능한데, `-primetext` 옵션을 사용하여 시작 텍스트를 설정할 수 있다. primtext는 직접 입력한 문자열을 이용하여 텍스트를 생성하기 전에, RNN이 특정 문맥에서 **준비**를 시작하도록 한다. 예를 들어 재미있는 primetext는 `-primetext "the meaning of life is "`이다.
> 주) 한글 데이터셋인 발라드 가사의 경우 `-primetext "사랑이 "`다.

**GPU로 훈련 후 CPU로 sampling**. 현재로써 해결방법은 `convert_gpu_cpu_checkpoint.lua` 스크립트를 사용하여 GPU 체크포인트를 CPU 체크포인트로 변환하는 것이다. 앞으로는 이러한 변환을 명시적으로 하지 않아도 되도록 수정할 예정이다. 예들 들어:

```
$ th convert_gpu_cpu_checkpoint.lua cv/lm_lstm_epoch30.00_1.3950.t7
```

는 새로운 `cv/lm_lstm_epoch30.00_1.3950.t7_cpu.t7` 파일을 생성하며, `-gpuid -1`를 통해 CPU 모드에서 샘플링 스크립트를 사용할 수 있다.

샘플링을 즐겨보시길!

## 팁과 트릭

### 평가 손실 과 훈련 손실 모니터링

머신 러닝과 신경망을 처음 접한다면, 좋은 모델을 만들기까지는 전문지식을 꽤 많이 습득해야 한다. 정량화된 값 중 계속 파악해야 하는 수치는 훈련 손실(훈련 단계에서 출력)과 평가 손실(RNN이 평가 데이터에 대해 실행될 때마다 출력(기본값은 1000 이터레이션))이다. 무엇보다도:

- 훈련 손실이 평가 손실보다 상당히 낮다면, 네트워크가 **과적합**되었다는 뜻이다. 해법은 네트워크 사이즈를 줄이거나, dropout을 높이는 것이다. 예를 들어 dropout을 0.5 등으로 설정해 볼 수 있다.
- 훈련/평가 손실이 거의 변함이 없다면 모델이 **부적합**되었다는 뜻이다. 이 경우에는 모델의 사이즈를 늘린다(또는 레이어 개수 또는 레이어당 뉴런 개수를 늘린다).

### 파라미터에 대한 근사치

모델을 제어하는 가장 중요한 두가지 파라미터는 `rnn_size`와 `num_layers`다. `num_layers`는 2 또는 3을 항상 사용할 것을 권한다. `rnn_size`은 학습할 데이터의 사이즈에 따라 달라진다. 눈여겨 봐야할 가장 중요한 2가지 측정값은

- 모델 파라미터의 개수. 훈련을 시작할 때 출력된다
- 데이터셋의 사이즈. 1MB는 대략 1M(백만)개의 문자에 해당한다.

이들 2가지 수치는 거의 동일한 자리수를 가져야 한다. 말로 표현하기는 않은데, 아래의 예를 보자:

- 100MB 데이터셋이 있고, 기본 파라미터 설정(현재 버전에서는 150K 정도의 파라미터)을 사용한다고 해보자. 데이터의 사이즈가 훨씬 크므로(100M >> 0.15M), 모델은 상당히 부적합하게 된다. 이 경우에는 간단히 `rnn_size`를 더 크게 만들면 되겠다.
- 10MB 데이터셋이 있고, 모델에서는 10M개의 파라미터를 사용한다고 해보자. 이 경우 조금 염려가 되므로 평가 손실을 주의깊게 살펴본다. 평가 손실이 훈련 손실보다 높다면, dropout을 조금 높여보고 평가 손실을 낯추는데 도움이 되는지 확인해 본다.

### 최고의 모델을 위한 전략

상당히 좋은 모델을 만들기 위한 성공 전략은 네트워크를 가능한한 최대의 크기로 만들고(학습하는데 걸리는 시간을 감내할 수 있을정도로) 다양한 dropout 값을 설정(0과 1사이의 값으로)하면서 의도적으로 실패과정을 겪는 것이다. 이 과정에서 평가 성능이 가장 좋은 모델(체크포인트 파일명에 사용된 평가 손실이 가장 낮은)을 얻게 된다면, 최종에는 이 모델을 사용할 수 있게 된다.

딥러닝의 경우 다양한 하이퍼파리미터 설정을 사용하여 서로 다른 모델을 실행해보는 것이 일반적이며, 최종적으로 가장 높은 평가 성능을 가지는 체크포인트를 취하게 된다.

덧붙여서, 훈련/평가 split 사이즈 또한 파라미터에 해당한다. 평가 데이터셋의 사이즈가 적절하도록 만들어야 하며, 그렇지 않을 경우 평가 성능은 신뢰할 수 없으며 그다지 의미있는 정보가 되지 못한다.

## 추가자료와 감사의 말

이 코드는 원래 옥스포드 대학교 머신 러닝 수업 [practical 6](https://github.com/oxford-cs-ml-2015/practical6)을 기반으로 했으며, 해당 코드는 또한 Wojciech Zaremba의 [learning to execute](https://github.com/wojciechz/learning_to_execute) 코드를 기반으로 한다. 그리고 코드의 일부분은 나의 연구실 동료인 [Justin Johnson](http://cs.stanford.edu/people/jcjohns/)와 함께 작업했다.

RNN 언어 모델에 대한 더 많은 자료를 보고 싶다면, 아래 자료를 볼 것을 권한다.

- [나의 최근 발표](https://skillsmatter.com/skillscasts/6611-visualizing-and-understanding-recurrent-networks) on char-rnn
- [Generating Sequences With Recurrent Neural Networks](http://arxiv.org/abs/1308.0850) by Alex Graves
- [Generating Text with Recurrent Neural Networks](http://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf) by Ilya Sutskever
- [Tomas Mikolov's Thesis](http://www.fit.vutbr.cz/~imikolov/rnnlm/thesis.pdf)

## 라이센스

MIT
