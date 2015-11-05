

- Dependencies
    - PCRE-8.37
    - http://www.linuxfromscratch.org/blfs/view/svn/general/pcre.html
    - `luarocks install Lrexlib-PCRE

    - loadcaffe
    - torchffi(?)

    - image, libjpeg
    - wget http://www.ijg.org/files/jpegsrc.v6b.tar.gz
    - tar -xvzf jpegsrc.v6b.tar.gz
    - cd jpeg-6b
    - ./configure --prefix=/home/abhshkdz/local
    - make CFLAGS=-fpic
    - make install / make install-lib

- Known issues
    - LuaJIT memory limit
    - Install Torch with plain Lua to avoid memory issues
    - Read more here: https://github.com/karpathy/char-rnn/issues/80#issuecomment-129791317


- npy4th
    - change to c++0x in CMakeLists.txt

- 200d GloVe embeddings
    - http://nlp.stanford.edu/projects/glove/
