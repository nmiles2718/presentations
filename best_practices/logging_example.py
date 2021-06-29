import logging

logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s:%(funcName)s:%(lineno)d]'
                           ' %(message)s')
LOG = logging.getLogger()
LOG.setLevel(logging.INFO)

LOG.info('info!')
LOG.warning('warning!')
LOG.error('error!')

def cool_function(input1=None, input2=None):
	
	try:
		msg = (
			f"\ninput1: {input1:0.3f}\n"
			f"input2: {input2}\n{'-'*70}"
		)
	except ValueError as e:
		# Show the warning
		LOG.error(e)
		msg = (
			f"\ninput1: {input1}\n"
			f"input2: {input2}\n{'-'*70}"
		)

	# Print the msg to console
	LOG.info(msg)

	try:
		input1 += 2
	except TypeError as e:
		LOG.error(e)
	else:
		LOG.info(f"Modified input 1: {input1:0.3f}")