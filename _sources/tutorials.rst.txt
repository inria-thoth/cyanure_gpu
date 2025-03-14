Tutorials
=========

.. note:: Many of the datasets I used are available `here <http://pascal.inrialpes.fr/data2/mairal/data/>`_.

For each dataset, we first give the result with the CPU version library and then the GPU one.

Examples for binary classification
----------------------------------
The following code performs binary classification with :math:`\ell_2`-regularized logistic regression, with no intercept, on the ocr dataset (23.1Gb)::

    from cyanure_gpu.estimators import Classifier
    from cyanure_gpu.data_processing import preprocess
    import numpy as np

    #load rcv1 dataset about 1Gb, n=781265, p=47152
    data = np.load(datapath + 'ocr.npz')
    y=np.squeeze(data['arr_1'])
    X=data['arr_0']

    #normalize the rows of X in-place, without performing any copy
    preprocess(X,normalize=True,columns=False)
    #declare a binary classifier for l2-logistic regression  uses the auto solver by default, performs at most 500 epochs
    classifier=Classifier(loss='logistic',penalty='l2',lambda_1=0.001953125,max_iter=500,tol=1e-3,duality_gap_interval=10, verbose=True, fit_intercept=False)
    classifier.fit(X,y)

Before we comment the previous choices, let us 
run the above code on one Intel(R) Xeon(R) Gold 6430 having access to 126 Go of RAM and a NVIDIA 6000 ADA GPU (48 Go of memory).  ::

    Info : Matrix X, n=2500000, p=1155
    Info : Catalyst Accelerator
    Info : ISTA Solver
    Info : Logistic Loss is used
    Info : L2 regularization
    Info : Epoch: 10, primal objective: 0.68547309280959156652, time: 32.47520999999999702368
    Info : Best relative duality gap: 0.03308400807599472249
    Info : Epoch: 20, primal objective: 0.67945171539393311999, time: 62.51010200000000338605
    Info : Best relative duality gap: 0.02137987285656091016
    Info : Epoch: 30, primal objective: 0.67547586698653583337, time: 92.48795599999999694774
    Info : Best relative duality gap: 0.01415023353528127754
    Info : Epoch: 40, primal objective: 0.67284701111919842376, time: 122.08335099999999329157
    Info : Best relative duality gap: 0.00939130161126555500
    Info : Epoch: 50, primal objective: 0.67110736628657929881, time: 150.75117199999999684223
    Info : Best relative duality gap: 0.00623283340952181155
    Info : Epoch: 60, primal objective: 0.66995531732809265879, time: 180.08448400000000333421
    Info : Best relative duality gap: 0.00413627674201270425
    Info : Epoch: 70, primal objective: 0.66919185798389579922, time: 208.72120499999999765350
    Info : Best relative duality gap: 0.00274527966941450624
    Info : Epoch: 80, primal objective: 0.66868557141165807511, time: 238.07016100000001301851
    Info : Best relative duality gap: 0.00182258662660126974
    Info : Epoch: 90, primal objective: 0.66834960668510767778, time: 267.40909900000002608067
    Info : Best relative duality gap: 0.00121049088143456362
    Info : Epoch: 100, primal objective: 0.66812652198281208271, time: 296.27869800000001987428
    Info : Best relative duality gap: 0.00080433396431857948
    Info : Time elapsed : 297.86861499999997704435


    2024-11-26 15:39:07,288 - INFO - simple_erm - Matrix X, n=2500000, p=1155
    2024-11-26 15:39:07,314 - INFO - accelerator - Catalyst Accelerator
    2024-11-26 15:39:07,314 - INFO - ista - ISTA Solver
    2024-11-26 15:39:07,400 - INFO - solver - *********************************
    2024-11-26 15:39:07,401 - INFO - logistic - Logistic Loss is used
    2024-11-26 15:39:07,401 - INFO - ridge - L2 regularization
    2024-11-26 15:39:08,458 - INFO - solver - Epoch: 10, primal objective: tensor([0.68547308444976806641], device='cuda:0'), time: 0.92639
    2024-11-26 15:39:08,491 - INFO - solver - Best relative duality gap: tensor([0.03308409452438354492], device='cuda:0')
    2024-11-26 15:39:09,142 - INFO - solver - Epoch: 20, primal objective: tensor([0.67945176362991333008], device='cuda:0'), time: 1.81144
    2024-11-26 15:39:09,174 - INFO - solver - Best relative duality gap: tensor([0.02137989178299903870], device='cuda:0')
    2024-11-26 15:39:09,826 - INFO - solver - Epoch: 30, primal objective: tensor([0.67547589540481567383], device='cuda:0'), time: 2.49520
    2024-11-26 15:39:09,858 - INFO - solver - Best relative duality gap: tensor([0.01415032148361206055], device='cuda:0')
    2024-11-26 15:39:10,695 - INFO - solver - Epoch: 40, primal objective: tensor([0.67284703254699707031], device='cuda:0'), time: 3.36445
    2024-11-26 15:39:10,727 - INFO - solver - Best relative duality gap: tensor([0.00939132738858461380], device='cuda:0')
    2024-11-26 15:39:11,378 - INFO - solver - Epoch: 50, primal objective: tensor([0.67110735177993774414], device='cuda:0'), time: 4.04747
    2024-11-26 15:39:11,411 - INFO - solver - Best relative duality gap: tensor([0.00623279577121138573], device='cuda:0')
    2024-11-26 15:39:12,062 - INFO - solver - Epoch: 60, primal objective: tensor([0.66995537281036376953], device='cuda:0'), time: 4.73148
    2024-11-26 15:39:12,096 - INFO - solver - Best relative duality gap: tensor([0.00413630390539765358], device='cuda:0')
    2024-11-26 15:39:12,747 - INFO - solver - Epoch: 70, primal objective: tensor([0.66919183731079101562], device='cuda:0'), time: 5.41596
    2024-11-26 15:39:12,779 - INFO - solver - Best relative duality gap: tensor([0.00274521391838788986], device='cuda:0')
    2024-11-26 15:39:13,563 - INFO - solver - Epoch: 80, primal objective: tensor([0.66868561506271362305], device='cuda:0'), time: 6.23248
    2024-11-26 15:39:13,595 - INFO - solver - Best relative duality gap: tensor([0.00182258465792983770], device='cuda:0')
    2024-11-26 15:39:14,247 - INFO - solver - Epoch: 90, primal objective: tensor([0.66834962368011474609], device='cuda:0'), time: 6.91590
    2024-11-26 15:39:14,279 - INFO - solver - Best relative duality gap: tensor([0.00121046497952193022], device='cuda:0')
    2024-11-26 15:39:14,930 - INFO - solver - Epoch: 100, primal objective: tensor([0.66812652349472045898], device='cuda:0'), time: 7.59990
    2024-11-26 15:39:14,963 - INFO - solver - Best relative duality gap: tensor([0.00080433191033080220], device='cuda:0')
    2024-11-26 15:39:14,963 - INFO - solver - This is the elapsed time: 7.599904775619507


The solver used was *catalyst-ista*; the problem was solved up to
accuracy tol=0.001 in about 7.6 sec after 100 epochs (without taking into account
the time to load the dataset from the hard drive). The regularization
parameter is the one given by cross-validation. We can see that the GPU version is about 40 times faster.

In the next example, we use the logistic loss with
:math:`\ell_1`-regularization, the regularization parameter is such that the
obtained solution has 0.1\% non-zero coefficients. 
We also fit an intercept.::

    from cyanure_gpu.estimators import Classifier
    from cyanure_gpu.data_processing import preprocess
    import numpy as np

    #load rcv1 dataset about 1Gb, n=781265, p=47152
    data = np.load(datapath + 'epsilon.npz')
    y=np.squeeze(data['arr_1'])
    X=data['arr_0']

    #normalize the rows of X in-place, without performing any copy
    preprocess(X,normalize=True,columns=False)
    #declare a binary classifier for squared hinge loss + l1 regularization
    classifier=Classifier(loss='logistic',penalty='l1',lambda_1=0.000244140625 * X.shape[0],max_iter=500,tol=1e-3, duality_gap_interval=10, verbose=True, fit_intercept=True)
    # uses the auto solver by default, performs at most 500 epochs
    classifier.fit(X,y) 

which yields::

    Info : Matrix X, n=250000, p=2000
    Info : Catalyst Accelerator
    Info : ISTA Solver
    Info : Logistic Loss is used
    Info : L1 regularization
    Info : Epoch: 10, primal objective: 0.69314561929310380961, time: 5.66298200000000040433
    Info : Best relative duality gap: 0.00000015295459037478
    Info : Time elapsed : 5.92307700000000014739


    2024-11-22 11:09:07,709 - INFO - simple_erm - Matrix X, n=250000, p=2000
    2024-11-22 11:09:07,782 - INFO - accelerator - Catalyst Accelerator
    2024-11-22 11:09:07,782 - INFO - ista - ISTA Solver
    2024-11-22 11:09:07,798 - INFO - solver - *********************************
    2024-11-22 11:09:07,798 - INFO - logistic - Logistic Loss is used
    2024-11-22 11:09:07,798 - INFO - lasso - L1 regularization
    2024-11-22 11:09:08,029 - INFO - solver - Epoch: 10, primal objective: tensor([0.69314563274383544922], device='cuda:0'), time: 0.20345
    2024-11-22 11:09:08,074 - INFO - solver - Best relative duality gap: tensor([1.71983032259959145449e-07], device='cuda:0')
    2024-11-22 11:09:08,074 - INFO - solver - This is the elapsed time: 0.2034461498260498



Multiclass classification
-------------------------
Let us now do something a bit more involved and perform multinomial logistic regression on the
*ckn_mnist* dataset (10 classes, n=60000, p=2304, dense matrix), with :math:`\ell_2` regularization,
still using an Intel(R) Xeon(R) Gold 6430 having access to 126 Go of RAM and a NVIDIA 6000 ADA GPU.::

    from cyanure_gpu.estimators import Classifier
    from cyanure_gpu.data_processing import preprocess
    import numpy as np

    #load ckn_mnist dataset 10 classes, n=60000, p=2304
    data=np.load(datapath + 'ckn_mnist.npz')
    y=data['y']
    y = np.squeeze(y)
    X=data['X']

    #center and normalize the rows of X in-place, without performing any copy
    preprocess(X,centering=True,normalize=True,columns=False)
    #declare a multinomial logistic classifier with group Lasso regularization
    classifier=Classifier(loss='multiclass-logistic',penalty='l2', solver='qning-ista', lambda_1=0.0009765625*X.shape[0],max_iter=500,tol=1e-15,duality_gap_interval=5, verbose=True, fit_intercept=False)
    # uses the auto solver by default, performs at most 500 epochs
    classifier.fit(X,y)

which produces::

    Info : Matrix X, n=60000, p=2304
    Info : Memory parameter: 20
    Info : QNing Accelerator
    Info : ISTA Solver
    Info : Multiclass logistic Loss is used
    Info : L2 regularization
    Info : Epoch: 5, primal objective: 1.34660935401916503906, time: 1.07866200000000000969
    Info : Best relative duality gap: 0.40878400206565856934
    Info : Epoch: 10, primal objective: 1.17538738250732421875, time: 1.93717999999999990202
    Info : Best relative duality gap: 0.01453427784144878387
    Info : Epoch: 15, primal objective: 1.16761827468872070312, time: 2.79002499999999997726
    Info : Best relative duality gap: 0.00151061406359076500
    Info : Epoch: 20, primal objective: 1.16685760021209716797, time: 3.64184000000000018815
    Info : Best relative duality gap: 0.00085969886276870966
    Info : Epoch: 25, primal objective: 1.16679251194000244141, time: 4.50452000000000030155
    Info : Best relative duality gap: 0.00001185153087135404
    Info : Epoch: 30, primal objective: 1.16678476333618164062, time: 5.46062199999999986488
    Info : Best relative duality gap: 0.00000143036663757812
    Info : Epoch: 35, primal objective: 1.16678380966186523438, time: 6.96293999999999968509
    Info : Best relative duality gap: -0.00000398459633288439
    Info : Time elapsed : 7.04140799999999966730
    Info : Total additional line search steps: 1
    Info : Total skipping l-bfgs steps: 2

    2024-11-21 17:27:16,101 - INFO - multi_erm - Matrix X, n=60000, p=2304
    2024-11-21 17:27:16,158 - INFO - accelerator - QNing Accelerator
    2024-11-21 17:27:16,158 - INFO - ista - ISTA Solver
    2024-11-21 17:27:16,219 - INFO - accelerator - Memory parameter: 20
    2024-11-21 17:27:16,227 - INFO - solver - *********************************
    2024-11-21 17:27:16,227 - INFO - multi_class_logistic - Multiclass logistic Loss is used
    2024-11-21 17:27:16,227 - INFO - ridge - L2 regularization
    2024-11-21 17:27:16,708 - INFO - solver - Epoch: 5, primal objective: tensor([1.34660909712377696579], device='cuda:0'), time: 0.50826
    2024-11-21 17:27:16,776 - INFO - solver - Best relative duality gap: tensor([0.40876695758362174837], device='cuda:0')
    2024-11-21 17:27:16,918 - INFO - solver - Epoch: 10, primal objective: tensor([1.17538727249934416008], device='cuda:0'), time: 0.75384
    2024-11-21 17:27:16,929 - INFO - solver - Best relative duality gap: tensor([0.01456019001919470722], device='cuda:0')
    2024-11-21 17:27:17,074 - INFO - solver - Epoch: 15, primal objective: tensor([1.16761827068007839614], device='cuda:0'), time: 0.90962
    2024-11-21 17:27:17,085 - INFO - solver - Best relative duality gap: tensor([0.00151743933299837062], device='cuda:0')
    2024-11-21 17:27:17,232 - INFO - solver - Epoch: 20, primal objective: tensor([1.16685757526350597502], device='cuda:0'), time: 1.06818
    2024-11-21 17:27:17,244 - INFO - solver - Best relative duality gap: tensor([0.00086651061339404446], device='cuda:0')
    2024-11-21 17:27:17,394 - INFO - solver - Epoch: 25, primal objective: tensor([1.16679242465549570795], device='cuda:0'), time: 1.22931
    2024-11-21 17:27:17,405 - INFO - solver - Best relative duality gap: tensor([3.85046781787790604848e-05], device='cuda:0')
    2024-11-21 17:27:17,576 - INFO - solver - Epoch: 30, primal objective: tensor([1.16678509730836377223], device='cuda:0'), time: 1.41183
    2024-11-21 17:27:17,587 - INFO - solver - Best relative duality gap: tensor([5.21347885587373416951e-06], device='cuda:0')
    2024-11-21 17:27:17,730 - INFO - solver - Epoch: 35, primal objective: tensor([1.16678362581394057251], device='cuda:0'), time: 1.56620
    2024-11-21 17:27:17,741 - INFO - solver - Best relative duality gap: tensor([3.20806198834368214258e-07], device='cuda:0')
    2024-11-21 17:27:17,885 - INFO - solver - Epoch: 40, primal objective: tensor([1.16678349137343806419], device='cuda:0'), time: 1.72061
    2024-11-21 17:27:17,896 - INFO - solver - Best relative duality gap: tensor([3.28692449350012228603e-08], device='cuda:0')
    2024-11-21 17:27:18,039 - INFO - solver - Epoch: 45, primal objective: tensor([1.16678347677815530403], device='cuda:0'), time: 1.87507
    2024-11-21 17:27:18,050 - INFO - solver - Best relative duality gap: tensor([3.18125686726953981710e-09], device='cuda:0')
    2024-11-21 17:27:18,194 - INFO - solver - Epoch: 50, primal objective: tensor([1.16678347529696058160], device='cuda:0'), time: 2.02985
    2024-11-21 17:27:18,205 - INFO - solver - Best relative duality gap: tensor([2.89937497949730604950e-10], device='cuda:0')
    2024-11-21 17:27:18,349 - INFO - solver - Epoch: 55, primal objective: tensor([1.16678347518372782510], device='cuda:0'), time: 2.18473
    2024-11-21 17:27:18,359 - WARNING - solver - The solution does not improve anymore, solving has been stopped.
    2024-11-21 17:27:18,360 - INFO - solver - Best relative duality gap: tensor([3.60835205423188893144e-11], device='cuda:0')
    2024-11-21 17:27:18,360 - INFO - solver - This is the elapsed time: 2.1847290992736816
    2024-11-21 17:27:18,360 - INFO - accelerator - Total additional line search steps: 1
    2024-11-21 17:27:18,360 - INFO - accelerator - Total skipping l-bfgs steps: 2





Learning the multiclass classifier took about 2.2s. To conclude, we provide a last more classical example
of learning l2-logistic regression classifiers on the same dataset, in a one-vs-all fashion. 
We notice that the CPU version greatly benifits from the number of cores which allows to parralelize all the solvers. 
That's why the GPU version is more than twice solver.::

    from cyanure_gpu.estimators import Classifier
    from cyanure_gpu.data_processing import preprocess
    import numpy as np

    #load ckn_mnist dataset 10 classes, n=60000, p=2304
    data=np.load(datapath + 'ckn_mnist.npz')
    y=data['y']
    y = np.squeeze(y)
    X=data['X']

    #center and normalize the rows of X in-place, without performing any copy
    preprocess(X,centering=True,normalize=True,columns=False)
    #declare a multinomial logistic classifier with group Lasso regularization
    classifier=Classifier(loss='logistic',penalty='l2',lambda_1=58.59375,max_iter=500,tol=1e-5,duality_gap_interval=10, multi_class="ovr",verbose=True, fit_intercept=False)
    # uses the auto solver by default, performs at most 500 epochs
    classifier.fit(X,y)


Then, the 10 classifiers are learned in parallel using the 2 CPUs, which gives the following output after about 18 sec::
    
    Info : Matrix X, n=60000, p=2304
    Info : Solver 8 has terminated after 20.00000000000000000000 epochs in 10.96056175231933593750 seconds
    Info :    Primal objective: 0.24975199997425079346, relative duality gap: -0.00000107394896531332
    Info : Solver 0 has terminated after 20.00000000000000000000 epochs in 11.69469165802001953125 seconds
    Info :    Primal objective: 0.20580284297466278076, relative duality gap: -0.00000890581850399030
    Info : Solver 4 has terminated after 20.00000000000000000000 epochs in 12.47539710998535156250 seconds
    Info :    Primal objective: 0.22394967079162597656, relative duality gap: -0.00000459112152384478
    Info : Solver 7 has terminated after 20.00000000000000000000 epochs in 15.23751926422119140625 seconds
    Info :    Primal objective: 0.17513881623744964600, relative duality gap: 0.00000059557402209975
    Info : Solver 3 has terminated after 20.00000000000000000000 epochs in 15.92408370971679687500 seconds
    Info :    Primal objective: 0.20927692949771881104, relative duality gap: -0.00000583865221415181
    Warning : Your problem is prone to numerical instability. It would be safer to use double.
    Info : Solver 2 has terminated after 50.00000000000000000000 epochs in 29.63804054260253906250 seconds
    Info :    Primal objective: 0.22416828572750091553, relative duality gap: 0.00007252215436892584
    Warning : Your problem is prone to numerical instability. It would be safer to use double.
    Info : Solver 5 has terminated after 70.00000000000000000000 epochs in 35.79794311523437500000 seconds
    Info :    Primal objective: 0.18814966082572937012, relative duality gap: 0.00001275095019082073
    Warning : Your problem is prone to numerical instability. It would be safer to use double.
    Info : Solver 6 has terminated after 70.00000000000000000000 epochs in 36.60394668579101562500 seconds
    Info :    Primal objective: 0.19530247151851654053, relative duality gap: 0.00038812740240246058
    Warning : Your problem is prone to numerical instability. It would be safer to use double.
    Info : Solver 1 has terminated after 80.00000000000000000000 epochs in 39.07707595825195312500 seconds
    Info :    Primal objective: 0.13079862296581268311, relative duality gap: 0.00001492410319769988
    Warning : Your problem is prone to numerical instability. It would be safer to use double.
    Info : Solver 9 has terminated after 90.00000000000000000000 epochs in 39.81833648681640625000 seconds
    Info :    Primal objective: 0.25878179073333740234, relative duality gap: 0.00001071024325938197
    Info : Time for the one-vs-all strategy
    Info : Time elapsed : 39.84983400000000131058



    2024-11-21 15:48:18,836 - INFO - multi_erm - Matrix X, n=60000, p=2304
    2024-11-21 15:48:27,789 - INFO - multi_erm - Solver 0 has terminated after 500.0 epochs in 8.92084789276123 seconds
    2024-11-21 15:48:27,790 - INFO - multi_erm -    Primal objective: 0.20580457427388008, relative duality gap: 9.707249846837206e-06
    2024-11-21 15:48:36,610 - INFO - multi_erm - Solver 1 has terminated after 500.0 epochs in 8.775974750518799 seconds
    2024-11-21 15:48:36,610 - INFO - multi_erm -    Primal objective: 0.1307992765715999, relative duality gap: 1.2421890926748529e-05
    2024-11-21 15:48:45,379 - INFO - multi_erm - Solver 2 has terminated after 500.0 epochs in 8.723823070526123 seconds
    2024-11-21 15:48:45,379 - INFO - multi_erm -    Primal objective: 0.22417289945725252, relative duality gap: 2.7430488492892824e-05
    2024-11-21 15:48:54,162 - INFO - multi_erm - Solver 3 has terminated after 500.0 epochs in 8.736879825592041 seconds
    2024-11-21 15:48:54,163 - INFO - multi_erm -    Primal objective: 0.20928146978693712, relative duality gap: 2.208283650841875e-05
    2024-11-21 15:49:02,977 - INFO - multi_erm - Solver 4 has terminated after 500.0 epochs in 8.773399114608765 seconds
    2024-11-21 15:49:02,977 - INFO - multi_erm -    Primal objective: 0.22395336127456733, relative duality gap: 2.3133542221329766e-05
    2024-11-21 15:49:11,773 - INFO - multi_erm - Solver 5 has terminated after 500.0 epochs in 8.750984191894531 seconds
    2024-11-21 15:49:11,773 - INFO - multi_erm -    Primal objective: 0.18815317301783568, relative duality gap: 2.5039743133227696e-05
    2024-11-21 15:49:20,538 - INFO - multi_erm - Solver 6 has terminated after 500.0 epochs in 8.718018531799316 seconds
    2024-11-21 15:49:20,539 - INFO - multi_erm -    Primal objective: 0.19528435559928414, relative duality gap: 1.3474321095759592e-05
    2024-11-21 15:49:29,362 - INFO - multi_erm - Solver 7 has terminated after 500.0 epochs in 8.782743215560913 seconds
    2024-11-21 15:49:29,362 - INFO - multi_erm -    Primal objective: 0.17514023007756355, relative duality gap: 1.2594092475734544e-05
    2024-11-21 15:49:38,157 - INFO - multi_erm - Solver 8 has terminated after 500.0 epochs in 8.750588417053223 seconds
    2024-11-21 15:49:38,158 - INFO - multi_erm -    Primal objective: 0.24975907330736724, relative duality gap: 3.1220033296235936e-05
    2024-11-21 15:49:46,923 - INFO - multi_erm - Solver 9 has terminated after 500.0 epochs in 8.72121000289917 seconds
    2024-11-21 15:49:46,923 - INFO - multi_erm -    Primal objective: 0.25878863035262567, relative duality gap: 3.7129335857905706e-05
    2024-11-21 15:49:46,924 - INFO - multi_erm - Time for the one-vs-all strategy
    2024-11-21 15:49:46,924 - INFO - multi_erm - Elapsed time: 88.08702564239502






