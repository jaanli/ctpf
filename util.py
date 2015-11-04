import logging

logger = logging.getLogger(__name__)

def calculate_loglikelihood(coder, train, validation, test):
  logger.info('=> calculating final log-likelihoods')
  train_ll = coder.pred_loglikeli(**train)
  logging.info('train ll:\t {0:.4f}'.format(train_ll))
  validation_ll = coder.pred_loglikeli(**validation)
  logging.info('validation ll:\t {0:.4f}'.format(validation_ll))
  test_ll = coder.pred_loglikeli(**test)
  logging.info('test ll:\t {0:.4f}'.format(test_ll))