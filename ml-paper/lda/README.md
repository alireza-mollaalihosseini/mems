this folder is supposed to use Linear Discriminant Analysis (LDA) method for both with and without sensor in audio scene classification.

the results show slight improvement (3-5%) in both cases.

here the is also another comparison between fft method and wavelet method where both classified with LDA. intrestingly, the results for wavelet were 77 and 76 (test, train). however, the results for fft method were 73 and 98 (test, train). this tells us that first of all, wavelet method gives higher testing accuracy with LDA (more robust) with lower columns in state matrix. secondly, based on a comparison between test and training, the fft method seems to overfit the task.