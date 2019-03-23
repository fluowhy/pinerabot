# PineraBot

Hello! I am PineraBot [![alt text][1.1]][1]. Welcome to my humble home.

## Table of contents
1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Usage](#usage)
3. [To do](#todo)
4. [Acknowledgments](#ack)

## Requirements <a name="requirements"></a>

* Python 3.6
* request
* contextlib2
* beautifulsoup4
* numpy
* sklearn
* unidecode
* pdb
* pytorch 1.0.x
* argparse
* tqdm

## Installation <a name="installations"></a>

Clone the repository:
```
git clone https://github.com/fluowhy/pinerabot.git
```
Install the requirements. Be careful with the PyTorch version, the proyect uses PyTorch 1.0.2 with Cuda support but you can use cpu support as well. The easiest way to install the requirements is with pip:
```
pip install <library name>
``` 
We suggest to make a [virtual environment](https://virtualenv.pypa.io/en/latest/) to manage the libraries.

## Usage <a name="usage"></a>

## To do <a name="todo"></a>

- [ ] Add Usage section.
- [ ] Check model capacity.
- [ ] Add **regularization**.
- [x] Code twitter bot.
- [x] Automatize tweets.
- [ ] Automatize when to tweet.
- [ ] Try bidirectional lstm.
- [ ] Add requirements file.
- [ ] Add conda environments option.
- *[x]* Add description.
- [ ] Clean code.
- [ ] **Make all numbers a single class**.
- [ ] **Solve labels problem**.

## Acknowledgments <a name="ack"></a>

* carlsednaoui [![alt text][1.2]][2] for gitsocial [![alt text][1.2]][3].

[1.1]: http://i.imgur.com/wWzX9uB.png
[1]: https://twitter.com/BotPinera
[1.2]: http://i.imgur.com/9I6NRUm.png
[2]: https://github.com/carlsednaoui
[3]: https://github.com/carlsednaoui/gitsocial