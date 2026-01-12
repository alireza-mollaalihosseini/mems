this folder is supposed to use Linear Discriminant Analysis (LDA) method for both with and without sensor in audio scene classification.

the results show slight improvement (3-5%) in both cases.

here there is also another comparison between fft method and wavelet method where both classified with LDA. intrestingly, the results for wavelet were 77 and 76 (test, train). however, the results for fft method were 73 and 98 (test, train). this tells us that first of all, wavelet method gives higher testing accuracy with LDA (more robust) with lower columns in state matrix. secondly, based on a comparison between test and training, the fft method seems to overfit the task.

but for the case with sensor (and LDA), the wavelet gives 49, 50 (test, train), while the fft gives 74, 99. in this case the testing with fft gives better results, while still gives the gap between train and test.

another comparison also applied here with having multiple sensors (36). the results for this implementaton were 45, 100 (test, train). In this task, the wavelet with LDA (also with timeseries features) were used, and as the results show, there is a very larg gap (overfiting).