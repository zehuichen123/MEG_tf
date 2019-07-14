## MEG(Maximum Entropy Generators for Energy-Based Models)

Here is my tensorflow implementation of MEG (<a href="https://arxiv.org/abs/1901.08508">https://arxiv.org/abs/1901.08508</a>, NIPS 2019 Under Review).

Note that MEG here is only implemented for anomaly detection.

There are still some bugs in this code, since the generator loss and E_fake loss here are increasing continuously, while in the MEG-pytorch code, the losses should be vibrated dramatically. And as the training goes, the gradient penalty here is prone to be the same. If you find the reason, please issue it, Thanks! 

### Reference

[1] MEG-pytorch (<a href="https://github.com/ritheshkumar95/energy_based_generative_models/blob/master/data/kdd.py">https://github.com/ritheshkumar95/energy_based_generative_models/blob/master/data/kdd.py</a>)