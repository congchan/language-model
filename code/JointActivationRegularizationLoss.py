from mxnet import gluon

class JointActivationRegularizationLoss(gluon.loss.Loss):
    r"""Computes Joint Regularization Loss with standard loss.

    The activation regularization refer to
    gluonnlp.loss.ActivationRegularizationLoss.

    The temporal activation regularization refer to
    gluonnlp.loss.TemporalActivationRegularizationLoss.

    Parameters
    ----------
    loss : gluon.loss.Loss
        The standard loss
    ar_loss: gluonnlp.loss.ActivationRegularizationLoss
        The activation regularization
    tar_loss: gluonnlp.loss.TemporalActivationRegularizationLoss
        The temporal activation regularization

    Inputs:
        - **out**: NDArray
        output tensor with shape `(sequence_length, batch_size, input_size)`
          when `layout` is "TNC".
        - **target**: NDArray
        target tensor with shape `(sequence_length, batch_size, input_size)`
          when `layout` is "TNC".
        - **states**: the stack outputs from RNN,
        which consists of output from each time step (TNC).
        - **dropped_states**: the stack outputs from RNN with dropout,
        which consists of output from each time step (TNC).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
          batch_axis are averaged out.
    """

    def __init__(self, l, ar_l, tar_l, weight=None, batch_axis=None, **kwargs):
        super(JointActivationRegularizationLoss, self).__init__(weight, batch_axis, **kwargs)
        self._loss = l
        self._ar_loss = ar_l
        self._tar_loss = tar_l

    def __repr__(self):
        s = 'JointActivationTemporalActivationRegularizationLoss'
        return s

    def hybrid_forward(self, F, out, target, states, dropped_states): # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        l = self._loss(out.reshape(-3, -1), target.reshape(-1,))
        l = l + self._ar_loss(*dropped_states)
        l = l + self._tar_loss(*states)
        return l
